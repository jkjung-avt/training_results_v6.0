# Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

import os
from math import ceil

import hydra
import megatron.bridge.training.setup as setup_module
import numpy as np
import torch
import utils
from callback_debug import StatsLogCallback
from callback_logging import DeltaTimingCallback, MLPerfLoggingCallback, mllogger
from callback_nvfp4 import NVFP4Callback
from callback_warmup import NsysProfileCallback, WarmupCallback
from megatron.bridge.data.packing import PackedSequenceSpecs
from megatron.bridge.models.gpt_provider import GPTModelProvider
from megatron.bridge.peft.lora import LoRA
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    DistributedInitConfig,
    GPTSFTDatasetConfig,
    LoggerConfig,
    OptimizerConfig,
    ProfilingConfig,
    RerunStateMachineConfig,
    RNGConfig,
    SchedulerConfig,
    TrainingConfig,
    ValidationConfig,
)
from megatron.bridge.training.finetune import finetune
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig
from megatron.bridge.training.tokenizers.config import TokenizerConfig


@hydra.main(version_base=None, config_path="conf", config_name="megatron_gpt_peft_tuning_config")
def main(cfg):
    if utils.get_rank() == 0:
        mllogger.start(key=mllogger.constants.INIT_START)
        mllogger.mlperf_submission_log(benchmark="llama2_70b_lora", num_nodes=cfg.trainer.num_nodes)
        mllogger.event(key="target_accuracy", value=0.925)

    cfg = utils.resolve(cfg)

    model_cfg = GPTModelProvider(
        # Llama 2 70B architecture
        hidden_size=8192,
        num_attention_heads=64,
        num_query_groups=8,
        ffn_hidden_size=28672,
        # Llama-specific defaults
        normalization="RMSNorm",
        activation_func=torch.nn.functional.silu,
        gated_linear_unit=True,
        position_embedding_type="rope",
        add_bias_linear=False,
        attention_dropout=0.0,
        hidden_dropout=0.0,
        share_embeddings_and_output_weights=False,
        bias_activation_fusion=True,
        persist_layer_norm=True,
        bias_dropout_fusion=True,
        apply_rope_fusion=True,
        # Parallelism settings
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        pipeline_model_parallel_size=cfg.model.pipeline_model_parallel_size,
        context_parallel_size=cfg.model.context_parallel_size,
        sequence_parallel=cfg.model.sequence_parallel,
        # Model architecture
        num_layers=cfg.model.num_layers,
        seq_length=cfg.model.encoder_seq_length,
        # Communication configuration
        cp_comm_type=cfg.model.cp_comm_type,
        calculate_per_token_loss=False,
        # Precision
        fp8_dot_product_attention=cfg.model.fp8_dot_product_attention,
        activation_func_fp8_input_store=cfg.model.activation_func_fp8_input_store,
        fp4_param=cfg.model.fp4_param,
        fp4=cfg.model.fp4,
        # Fusion optimizations
        cross_entropy_loss_fusion=cfg.model.cross_entropy_loss_fusion,
        cross_entropy_fusion_impl=cfg.model.cross_entropy_loss_fusion_impl,
        gradient_accumulation_fusion=cfg.model.gradient_accumulation_fusion,
        use_transformer_engine_op_fuser=cfg.model.use_transformer_engine_op_fuser,
        fused_single_qkv_rope=cfg.model.fused_single_qkv_rope,
        disable_parameter_transpose_cache=True,
        # CUDA Graphs
        cuda_graph_impl="local" if int(os.getenv("MCORE_CUDA_GRAPH", "0")) == 1 else "none",
        cuda_graph_scope=cfg.model.cuda_graph_scope,
        use_te_rng_tracker=cfg.model.use_te_rng_tracker,
        # CPU offloading
        cpu_offloading=cfg.model.cpu_offloading,
        cpu_offloading_weights=cfg.model.cpu_offloading_weights,
        cpu_offloading_num_layers=cfg.model.cpu_offloading_num_layers,
        cpu_offloading_activations=cfg.model.cpu_offloading_activations,
        cpu_offloading_double_buffering=cfg.model.cpu_offloading_double_buffering,
        # Recomputation
        recompute_method=cfg.model.recompute_method,
        recompute_modules=cfg.model.recompute_modules,
        recompute_num_layers=cfg.model.recompute_num_layers,
        recompute_granularity=cfg.model.recompute_granularity,
        distribute_saved_activations=cfg.model.distribute_saved_activations,
    )

    if utils.get_rank() == 0:
        mllogger.event(key="lora_rank", value=16)
        mllogger.event(key="lora_alpha", value=32)

    lora_config = LoRA(
        dim=16,
        alpha=32,
        dropout=0.1,
        a2a_experimental=True,
        dropout_position="pre",
        lora_A_init_method="kaiming",
        target_modules=["linear_proj", "linear_qkv"],
    )

    if not cfg.load_ckpt:
        setup_module._create_peft_pre_wrap_hook = utils.create_debug_peft_hook

    checkpoint_config = CheckpointConfig(
        pretrained_checkpoint=f"{cfg.ckpt_root}",
        dist_ckpt_strictness="log_all",
        fully_parallel_load=cfg.parallel_load,
        load_optim=False,
        load_rng=False,
        load_main_params_from_ckpt=False,
        finetune=True,
    )

    if utils.get_rank() == 0:
        mllogger.event(key=mllogger.constants.OPT_BASE_LR, value=cfg.optim.lr)
        mllogger.event(key=mllogger.constants.OPT_ADAMW_WEIGHT_DECAY, value=0.0001)
        mllogger.event(key=mllogger.constants.OPT_GRADIENT_CLIP_NORM, value=0.3)

    optimizer_config = OptimizerConfig(
        # General
        optimizer="adam",
        lr=cfg.optim.lr,
        min_lr=0,
        clip_grad=0.3,
        weight_decay=0.0001,
        adam_beta1=0.9,
        adam_beta2=0.999,
        adam_eps=1e-08,
        # Precision
        bf16=True,
        params_dtype=torch.bfloat16,
        # Distributed
        use_distributed_optimizer=cfg.optim.use_distributed_optimizer,
        overlap_param_gather_with_optimizer_step=cfg.optim.overlap_param_gather_with_optimizer_step,
    )

    if utils.get_rank() == 0:
        mllogger.event(
            key=mllogger.constants.OPT_LR_WARMUP_FACTOR,
            value=cfg.optim.sched.warmup_steps / cfg.trainer.max_steps,
        )

    scheduler_config = SchedulerConfig(
        lr_decay_style="cosine",
        start_weight_decay=0.0001,
        end_weight_decay=0.0001,
        lr_warmup_fraction=cfg.optim.sched.warmup_steps / cfg.trainer.max_steps,
    )

    if utils.get_rank() == 0:
        mllogger.event(
            key=mllogger.constants.OPT_LR_TRAINING_STEPS,
            value=cfg.trainer.max_steps,
        )

    if utils.get_rank() == 0:
        ga = int(os.getenv("MINIBS", "1")) // cfg.trainer.micro_batch_size
        mllogger.event(key=mllogger.constants.GRADIENT_ACCUMULATION_STEPS, value=ga)
        mllogger.event(
            key=mllogger.constants.GLOBAL_BATCH_SIZE,
            value=cfg.trainer.global_batch_size,
        )

    training_config = TrainingConfig(
        micro_batch_size=cfg.trainer.micro_batch_size,
        global_batch_size=cfg.trainer.global_batch_size,
        train_iters=cfg.trainer.max_steps,
        # GC Config
        manual_gc=True,
        manual_gc_interval=10000,
        manual_gc_eval=False,
        # Memory management
        empty_unused_memory_level=0,
        train_sync_interval=None,
        # Skip numeric checks
        check_optimizer_step_success=False,
        skip_sync_grad_norm_across_mp=True,
        exit_signal_handler=False,
    )

    if cfg.trainer.val_micro_batch_size is None:
        cfg.trainer.val_micro_batch_size = 1
    validation_config = ValidationConfig(
        eval_interval=cfg.trainer.val_check_interval,
        eval_iters=ceil(173 / cfg.trainer.val_global_batch_size),
        eval_global_batch_size=cfg.trainer.val_global_batch_size,
        eval_micro_batch_size=cfg.trainer.val_micro_batch_size,
        start_eval_at_iter=(cfg.skip_evals + 1) * cfg.trainer.val_check_interval,
    )

    data_parallel_sharding_strategy = "optim_grads_params" if cfg.model.fsdp == "megatron" else "no_shard"
    use_custom_fsdp = cfg.model.fsdp == "megatron"
    ddp_config = DistributedDataParallelConfig(
        # Communication overlap
        overlap_grad_reduce=cfg.ddp.overlap_grad_reduce,
        overlap_param_gather=cfg.ddp.overlap_param_gather,
        bucket_size=cfg.ddp.bucket_size,
        # FP8 parameter gathering
        fp8_param_gather=cfg.ddp.fp8_param_gather,
        # Gradient settings
        average_in_collective=cfg.ddp.average_in_collective,
        use_distributed_optimizer=cfg.optim.use_distributed_optimizer,
        # FSDP settings
        use_custom_fsdp=use_custom_fsdp,
        data_parallel_sharding_strategy=data_parallel_sharding_strategy,
        # NCCL settings
        nccl_ub=cfg.ddp.nccl_ub,
        fsdp_double_buffer=cfg.ddp.nccl_ub,
    )

    data_root = cfg.data_root
    if not os.path.exists(data_root):
        raise FileNotFoundError(f"Data root directory not found: {data_root}")

    train_path = f"{data_root}/train.npy"
    val_path = f"{data_root}/validation.npy"
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        raise FileNotFoundError(f"Training or validation data not found in {data_root}")

    if utils.get_rank() == 0:
        mllogger.event(key=mllogger.constants.TRAIN_SAMPLES, value=3901)
        mllogger.event(key=mllogger.constants.EVAL_SAMPLES, value=173)

    packed_sequence_specs = PackedSequenceSpecs(
        packed_sequence_size=cfg.model.encoder_seq_length,
        packed_train_data_path=f"{data_root}/train.npy",
        packed_val_data_path=f"{data_root}/validation.npy",
        packed_metadata_path=f"{data_root}/train_metadata.jsonl",
        pad_cu_seqlens=True,
    )

    dataset_config = GPTSFTDatasetConfig(
        dataloader_type="batch",
        dataset_root=data_root,
        seq_length=8192,
        seed=cfg.model.seed,
        do_validation=True,
        do_test=False,
        persistent_workers=True,
        num_workers=cfg.dataloader.num_workers,
        offline_packing_specs=packed_sequence_specs,
        enable_offline_packing=True,
        dataset_kwargs={
            "pad_to_max_length": True,
            "return_cu_seqlen": False,
        },
        max_train_samples=int(ceil(cfg.trainer.global_batch_size * cfg.trainer.max_steps * 1.005)),
    )

    fp8_type = "hybrid" if cfg.model.fp8 else None
    fp4_type = "e2m1" if cfg.model.fp4 else None
    mixed_precision = MixedPrecisionConfig(
        bf16=True,
        grad_reduce_in_fp32=False,
        params_dtype=torch.bfloat16,
        first_last_layers_bf16=cfg.model.first_last_layers_bf16,
        num_layers_at_start_in_bf16=cfg.model.num_layers_at_start_in_bf16,
        num_layers_at_end_in_bf16=cfg.model.num_layers_at_end_in_bf16,
        # fp4
        fp4=fp4_type,
        fp4_recipe=cfg.model.fp4_recipe,
        # fp8
        fp8=fp8_type,
        fp8_recipe=cfg.model.fp8_recipe,
        fp8_amax_history_len=cfg.model.fp8_amax_history_len,
        fp8_amax_compute_algo=cfg.model.fp8_amax_compute_algo,
        fp8_param_gather=cfg.model.fp8_param_gather,
        fp8_dot_product_attention=cfg.model.fp8_dot_product_attention,
    )

    dist_config = DistributedInitConfig(
        distributed_backend="nccl",
    )

    tokenizer_config = TokenizerConfig(
        tokenizer_type="HuggingFaceTokenizer",
        hf_tokenizer_kwargs={"use_fast": True},
        tokenizer_model=f"{cfg.ckpt_root}/iter_0000000/tokenizer",
    )

    if utils.get_rank() == 0:
        mllogger.event(key=mllogger.constants.SEED, value=cfg.model.seed, sync=False)

    rng_config = RNGConfig(
        seed=cfg.model.seed,
        te_rng_tracker=cfg.model.use_te_rng_tracker,
    )

    logger_config = LoggerConfig(
        log_interval=cfg.trainer.max_steps + 1,
        skip_train_metrics_log=True,
        timing_log_level=-1,
    )

    rerun_state_machine_config = RerunStateMachineConfig(
        rerun_mode="disabled",
        check_for_nan_in_loss=False,
        check_for_spiky_loss=False,
    )

    profile_ranks = [
        int(r) for r in str(cfg.nsys_profile.ranks_str).split(",") if str(cfg.nsys_profile.ranks_str).strip()
    ]
    profiling_config = ProfilingConfig(
        use_nsys_profiler=cfg.nsys_profile.enabled,
        profile_step_start=cfg.nsys_profile.start_step,
        profile_step_end=cfg.nsys_profile.end_step,
        profile_ranks=profile_ranks,
        nvtx_ranges=cfg.nsys_profile.nvtx_ranges,
    )

    config = ConfigContainer(
        rng=rng_config,
        ddp=ddp_config,
        model=model_cfg,
        dist=dist_config,
        peft=lora_config,
        logger=logger_config,
        train=training_config,
        validation=validation_config,
        dataset=dataset_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        tokenizer=tokenizer_config,
        checkpoint=checkpoint_config,
        mixed_precision=mixed_precision,
        rerun_state_machine=rerun_state_machine_config,
        profiling=profiling_config,
    )
    callbacks = []
    if cfg.model.fp4:
        callbacks.append(NVFP4Callback(cfg))
    if cfg.model.custom.warmup:
        callbacks.append(WarmupCallback(cfg, forward_step_func=utils.forward_step))
    callbacks.append(DeltaTimingCallback(cfg))
    callbacks.append(MLPerfLoggingCallback(cfg))
    if fname := os.environ.get("STAT_CALLBACK_FNAME"):
        callbacks.append(StatsLogCallback(save_path=fname))
    if cfg.nsys_profile.enabled:
        callbacks.append(
            NsysProfileCallback(
                start_step=cfg.nsys_profile.start_step,
                end_step=cfg.nsys_profile.end_step,
                profile_ranks=profile_ranks,
            )
        )

    finetune(config, utils.forward_step, callbacks=callbacks)


if __name__ == "__main__":
    main()
