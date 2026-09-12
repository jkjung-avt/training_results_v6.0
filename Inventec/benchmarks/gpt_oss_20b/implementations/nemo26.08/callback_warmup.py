# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import time
from pprint import pprint
from types import MethodType

import torch
import utils
from embedding_lib.custom_embedding.custom_embedding import CustomEmbedding
from megatron.bridge.data.samplers import build_pretraining_data_loader
from megatron.bridge.training import eval as eval_module
from megatron.bridge.training import train as train_module
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.training.tokenizers.config import TokenizerConfig
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.utils.train_utils import prepare_forward_step_func
from megatron.core import parallel_state
from megatron.core.datasets.blended_megatron_dataset_builder import (
    BlendedMegatronDatasetBuilder,
)
from megatron.core.datasets.gpt_dataset import MockGPTDataset
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from megatron.core.num_microbatches_calculator import get_num_microbatches
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.pipeline_parallel.p2p_communication import P2PCommunicator
from megatron.core.tensor_parallel.layers import VocabParallelEmbedding
from megatron.core.tensor_parallel.mappings import (
    reduce_from_tensor_model_parallel_region,
    reduce_scatter_to_sequence_parallel_region,
)
from megatron.core.transformer.cuda_graphs import TECudaGraphHelper
from megatron.core.transformer.enums import CudaGraphScope
from megatron.core.optimizer.distrib_optimizer import DistributedOptimizer
from megatron.core.utils import get_model_config

logger = logging.getLogger(__name__)


# Guard TECudaGraphHelper.delete_cuda_graphs against Bridge's ghost helper:
# Bridge's train.py builds its own TECudaGraphHelper when cuda_graph_impl ==
# "transformer_engine" but never captures on it (our WarmupCallback already did).
# MBridge #3459 (2026-04-22) made delete_cuda_graphs assert _graphs_created, which
# fires at teardown. Short-circuit the delete when no graphs were created.
_orig_te_delete_cuda_graphs = TECudaGraphHelper.delete_cuda_graphs


def _guarded_te_delete_cuda_graphs(self):
    if not self._graphs_created:
        return
    return _orig_te_delete_cuda_graphs(self)


TECudaGraphHelper.delete_cuda_graphs = _guarded_te_delete_cuda_graphs


def custom_embed_forward(self, input_):
    if self.tp_group.size() > 1:
        # Build the mask.
        input_mask = (input_ < self.vocab_start_index) | (input_ >= self.vocab_end_index)
        # Mask the input.
        masked_input = input_.clone() - self.vocab_start_index
        masked_input[input_mask] = 0
    else:
        masked_input = input_
    # Get the embeddings.
    output_parallel = CustomEmbedding.apply(self.weight, masked_input, self.num_embeddings_per_partition)
    # Mask the output embedding.
    if self.tp_group.size() > 1:
        output_parallel[input_mask, :] = 0.0

    if self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        output_parallel = output_parallel.transpose(0, 1).contiguous()
        output = reduce_scatter_to_sequence_parallel_region(output_parallel, group=self.tp_group)
    else:
        # Reduce across all the model parallel GPUs.
        if self.tp_group.size() > 1:
            output = reduce_from_tensor_model_parallel_region(output_parallel, group=self.tp_group)
        else:
            output = output_parallel
    return output


def create_mock_dataset_config(config):
    from megatron.bridge.training.config import MockGPTDatasetConfig

    vocab_size = config.model.data.mock_tokenizer_vocab_size
    tokenizer = build_tokenizer(
        config=TokenizerConfig(tokenizer_type="NullTokenizer", vocab_size=vocab_size),
        make_vocab_size_divisible_by=128,
        tensor_model_parallel_size=1,
    )

    mock_config = MockGPTDatasetConfig(
        seq_length=config.model.encoder_seq_length,
        random_seed=config.model.seed,
        dataloader_type="single",
        num_workers=config.model.data.num_workers,
        reset_position_ids=False,
        reset_attention_mask=False,
        eod_mask_loss=False,
        path_to_cache=None,
        split="900,50,50",
        data_sharding=True,
        create_attention_mask=False,
    )
    mock_config.tokenizer = tokenizer

    return mock_config



def capture_te_cuda_graph(context, model_config):
    """Capture TE CUDA graphs.

    The caller is responsible for managing forward pre-hook and param_sync_func
    state around this call — hooks should be disabled and param_sync_func should
    be None so that collectives are not baked into the captured graph.

    Returns the TECudaGraphHelper so the caller can set manual hooks after
    re-enabling forward pre-hooks.
    """
    cuda_graph_helper = TECudaGraphHelper(
        model=context.model,
        config=model_config,
        seq_length=context.state.cfg.model.seq_length,
        micro_batch_size=context.state.cfg.train.micro_batch_size,
        optimizers=[context.optimizer],
    )

    cuda_graph_helper.create_cudagraphs()

    # Prevent train.py from ever re-capturing
    model_config.cuda_graph_warmup_steps = -1
    return cuda_graph_helper


class WarmupCallback(Callback):
    def __init__(self, cfg, forward_step_func=None):
        self.cfg = cfg
        self.forward_step_func = forward_step_func
        self.train_steps = cfg.model.custom.warmup_train_steps
        self.val_steps = cfg.model.custom.warmup_validation_steps
        self.cuda_graph_helper = None
        self.optimizer_cuda_graph = cfg.model.optim.optimizer_cuda_graph

    def on_data_init_start(self, context: CallbackContext):
        self._optimizer = context.optimizer

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if utils.rank == 0:
            print("\nMCore config:", flush=True)
            pprint(context.model[0].config)

        for model_chunk in context.model:
            if hasattr(model_chunk.module.module, "embedding"):
                if isinstance(model_chunk.module.module.embedding.word_embeddings, VocabParallelEmbedding):
                    model_chunk.module.module.embedding.word_embeddings.__orig_forward__ = (
                        model_chunk.module.module.embedding.word_embeddings.forward
                    )
                    model_chunk.module.module.embedding.word_embeddings.forward = MethodType(
                        custom_embed_forward, model_chunk.module.module.embedding.word_embeddings
                    )

        torch.distributed.barrier()

        pg_collection = get_pg_collection(context.model)
        forward_backward_func = get_forward_backward_func(
            pp_size=pg_collection.pp.size(),
            vp_size=context.state.cfg.model.virtual_pipeline_model_parallel_size,
        )

        enable_cuda_graph = int(os.getenv("MCORE_CUDA_GRAPH", "0")) == 1
        cuda_graph_scope = context.state.cfg.model.cuda_graph_scope
        cg_warmup_steps = self.cfg.model.custom.cuda_graph_warmup_steps
        if enable_cuda_graph and self.cfg.model.nsys_profile.profile_cuda_graph_capture:
            # Inflate warmup steps so CG capture is deferred to the main training loop,
            # where it coincides with the nsys profiling window at profile_start_step.
            cg_warmup_steps = self.train_steps + self.cfg.model.nsys_profile.start_step
            if utils.rank == 0:
                logger.info(
                    f"profile_cuda_graph_capture enabled: setting cuda_graph_warmup_steps={cg_warmup_steps} "
                    f"(warmup_train_steps={self.train_steps} + "
                    f"profile_start_step={self.cfg.model.nsys_profile.start_step})"
                )
        if context.state.cfg.model.cuda_graph_impl == "local" and CudaGraphScope.full_iteration in cuda_graph_scope:
            forward_backward_func = FullCudaGraphWrapper(
                forward_backward_func,
                cuda_graph_warmup_steps=cg_warmup_steps,
            )

        prepared_forward_step_func = prepare_forward_step_func(self.forward_step_func, context.state)

        # Create and finalize mock dataset config
        mock_config = create_mock_dataset_config(self.cfg)
        mock_config.finalize()

        # Calculate number of samples needed for warmup
        micro_batch_size = context.state.cfg.train.micro_batch_size
        num_microbatches = get_num_microbatches()
        dp_size = parallel_state.get_data_parallel_world_size()

        # Samples needed: (steps * microbatches * micro_batch_size * dp_size)
        samples = (self.train_steps + self.val_steps) * num_microbatches * micro_batch_size * dp_size
        train_val_test_num_samples = [samples, 0, 0]

        # Build mock datasets using BlendedMegatronDatasetBuilder
        train_ds, _, _ = BlendedMegatronDatasetBuilder(
            MockGPTDataset,
            train_val_test_num_samples,
            lambda: True,  # is_dataset_built_on_rank - always True for mock
            mock_config,
        ).build()

        # Build dataloaders
        dataloader = build_pretraining_data_loader(
            dataset=train_ds,
            consumed_samples=0,
            dataloader_type=mock_config.dataloader_type or "single",
            micro_batch_size=micro_batch_size,
            num_workers=mock_config.num_workers,
            data_sharding=mock_config.data_sharding,
            data_parallel_rank=parallel_state.get_data_parallel_rank(),
            data_parallel_size=dp_size,
        )
        data_iterator = iter(dataloader)

        # Initialize pp group in warmup.
        pp_group = pg_collection.pp
        model_config = get_model_config(context.model[0])
        p2p_communicator = P2PCommunicator(pp_group=pp_group, config=model_config)
        torch.distributed.barrier(pp_group)

        # Mirror train.py: disable forward pre-hooks during warmup to prevent
        # stale data from propagating through the first all-gather.
        should_toggle_fwd_pre_hook = train_module.should_disable_forward_pre_hook(
            context.state.cfg.ddp.use_megatron_fsdp,
            context.state.cfg.optimizer.use_distributed_optimizer,
            context.state.cfg.ddp.overlap_param_gather,
        )
        if should_toggle_fwd_pre_hook:
            train_module.disable_forward_pre_hook(context.model, param_sync=False)
            param_sync_func = model_config.param_sync_func
            model_config.param_sync_func = None

        for group in context.optimizer.param_groups:
            group["betas_"] = group["betas"]
            group["bias_correction_"] = group["bias_correction"]
            group["lr_"] = group["lr"]
            group["weight_decay_"] = group["weight_decay"]
            group["betas"] = [1.0, 1.0]
            group["bias_correction"] = False
            group["lr"] = 0.0 if not self.optimizer_cuda_graph else torch.zeros_like(group["lr_"])
            group["weight_decay"] = 0.0

        scheduler_state = context.scheduler.state_dict()

        # Warmup for training
        if utils.rank == 0:
            logger.info("Starting training warmup")
        start = time.time()
        for step_idx in range(self.train_steps):
            if utils.rank == 0:
                logger.info(f"    Starting warmup step {step_idx}")
                step_timer = time.time()
            # Runs forward / backward step
            torch.cuda.synchronize()
            torch.distributed.barrier()
            train_module.train_step(
                forward_step_func=prepared_forward_step_func,
                data_iterator=data_iterator,
                model=context.model,
                optimizer=context.optimizer,
                scheduler=context.scheduler,
                global_state=context.state,
                pg_collection=pg_collection,
                forward_backward_func=forward_backward_func,
                p2p_communicator=p2p_communicator,
            )

            if step_idx == 0:
                # Capture CUDA graphs while hooks are still disabled and
                # param_sync_func is None (mirrors train.py L345 where
                # capture happens before hook enable at L438).
                if context.state.cfg.model.cuda_graph_impl == "transformer_engine":
                    self.cuda_graph_helper = capture_te_cuda_graph(context, context.state.cfg.model)
                # Enable forward pre-hooks after the first warmup step and
                # restore param_sync_func, mirroring train.py L438-441.
                if should_toggle_fwd_pre_hook:
                    train_module.enable_forward_pre_hook(context.model)
                    model_config.param_sync_func = param_sync_func
                    if self.cuda_graph_helper is not None:
                        self.cuda_graph_helper.cuda_graph_set_manual_hooks()

            torch.cuda.synchronize()
            torch.distributed.barrier()

            if utils.rank == 0:
                logger.info(f"    Finished warmup step {step_idx}, takes {time.time() - step_timer} s")

        if hasattr(context.scheduler, "num_steps"):
            context.scheduler.num_steps = 0
        context.scheduler.load_state_dict(scheduler_state)

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if utils.rank == 0:
            logger.info(f"Finished training warmup: {time.time() - start} s. ")

        # Recover optimizer configs changed by warmup
        for group in context.optimizer.param_groups:
            group["betas"] = group["betas_"]
            group["bias_correction"] = group["bias_correction_"]
            group["lr"] = group["lr_"]
            group["weight_decay"] = group["weight_decay_"]
            del group["betas_"]
            del group["bias_correction_"]
            del group["lr_"]
            del group["weight_decay_"]
            if "step" in group:
                if isinstance(group["step"], torch.Tensor):
                    group["step"].fill_(1)
                else:
                    group["step"] = 1

        # If no training steps ran, hooks are still disabled — enable now.
        if should_toggle_fwd_pre_hook and self.train_steps == 0:
            train_module.enable_forward_pre_hook(context.model)
            model_config.param_sync_func = param_sync_func

        # Warmup for validation
        if self.val_steps > 0:
            start_val_time = time.time()
            if utils.rank == 0:
                logger.info("Starting validation warmups")
            original_eval_iters = context.state.cfg.validation.eval_iters
            context.state.cfg.validation.eval_iters = self.val_steps

            # Populate param_data with correct param shards before validation so that
            # finish_param_sync (triggered by forward pre-hooks) all-gathers actual
            # weights instead of stale gradient data left over from the training step.
            # This mirrors the _handle_mxfp8_param_buffer_copy() call in train_step.
            if (
                context.state.cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag
                and context.state.cfg.ddp.overlap_param_gather
            ):
                for model_chunk in context.model:
                    model_chunk.zero_grad_buffer()
                for optim_instance in context.optimizer.chained_optimizers:
                    if isinstance(optim_instance, DistributedOptimizer):
                        optim_instance._copy_main_params_to_param_buffer()

            # Disable hooks before validation, mirroring train.py L517-518.
            if should_toggle_fwd_pre_hook:
                train_module.disable_forward_pre_hook(context.model)

            torch.cuda.synchronize()
            torch.distributed.barrier()
            eval_module.evaluate(
                context.state,
                prepared_forward_step_func,
                data_iterator,
                context.model,
                None,
                context.state.cfg,
                False,
            )

            context.state.train_state.consumed_valid_samples = 0
            context.state.cfg.validation.eval_iters = original_eval_iters

            # Re-enable hooks after validation, mirroring train.py L543-545.
            if should_toggle_fwd_pre_hook:
                train_module.enable_forward_pre_hook(context.model)

            torch.cuda.synchronize()
            torch.distributed.barrier()
            if utils.rank == 0:
                logger.info(f"Finished validation warmup: {time.time() - start_val_time} s. ")

        torch.cuda.synchronize()
        torch.distributed.barrier()
        if utils.rank == 0:
            logger.info(f"Time spent in run_training_warmup: {time.time() - start}s")

    def on_train_end(self, context: CallbackContext):
        # Release TE CUDA graphs captured during warmup. delete_cuda_graphs()
        # iterates callables_per_chunk, resets each graph, and clears
        # layer.cuda_graphs itself — a preceding manual loop would delete the
        # attribute and cause AttributeError inside delete_cuda_graphs().
        if self.cuda_graph_helper is None:
            return
        self.cuda_graph_helper.delete_cuda_graphs()
        self.cuda_graph_helper = None

    def on_eval_start(self, context: CallbackContext):
        # Populate param_data with correct param shards after validation so that
        # finish_param_sync (triggered by forward pre-hooks) all-gathers actual
        # weights instead of stale gradient data left over from the training step.
        # This mirrors the _handle_mxfp8_param_buffer_copy() call in train_step.
        if (
            context.state.cfg.optimizer.reuse_grad_buf_for_mxfp8_param_ag
            and context.state.cfg.ddp.overlap_param_gather
        ):
            for model_chunk in context.model:
                model_chunk.zero_grad_buffer()
            for optim_instance in self._optimizer.chained_optimizers:
                if isinstance(optim_instance, DistributedOptimizer):
                    optim_instance._copy_main_params_to_param_buffer()
            # disable_forward_pre_hook(param_sync=True) in train.py set
            # param_gather_dispatched=True via start_param_sync(force_sync=True).
            # zero_grad_buffer does NOT reset this flag. If it stays True, the next
            # training step's finish_param_sync becomes a no-op (skips the all-gather,
            # copy-to-param.data, and zero-param_data), corrupting both weights and
            # gradient buffers.
            for model_chunk in context.model:
                for bucket_group in (
                    model_chunk.bucket_groups + model_chunk.expert_parallel_bucket_groups
                ):
                    bucket_group.param_gather_dispatched = False
