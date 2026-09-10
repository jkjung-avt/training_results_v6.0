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

import logging
import os
from math import ceil, floor

import torch
from megatron.core import parallel_state
from omegaconf.omegaconf import OmegaConf


logging.getLogger("root").disabled = True
logging.getLogger("megatron.core.utils").disabled = True
logging.getLogger("megatron.core.optimizer_param_scheduler").setLevel(logging.WARNING)
logging.getLogger("megatron.core.distributed.param_and_grad_buffer").setLevel(logging.WARNING)
logging.getLogger("megatron.core.optimizer_param_scheduler.cosine_scheduler").disabled = True
logging.getLogger("megatron.bridge.training.utils.log_utils").setLevel(logging.WARNING)


rank = int(os.getenv("SLURM_PROCID", 0))


class RankZeroFilter(logging.Filter):
    def filter(self, record):
        return rank == 0


cg_logger = logging.getLogger("megatron.core.full_cuda_graph")
cg_logger.addFilter(RankZeroFilter())

torch.cuda.set_device(int(os.getenv("SLURM_LOCALID", "0")))


def force_all_tensors_to_non_fp8_patched(sharded_state_dict):
    return

import megatron.core.dist_checkpointing.utils as dist_checkpointing_utils
dist_checkpointing_utils.force_all_tensors_to_non_fp8 = force_all_tensors_to_non_fp8_patched
from importlib import import_module
module = import_module("megatron.core.dist_checkpointing.serialization")
current = getattr(module, "force_all_tensors_to_non_fp8")
setattr(module, "force_all_tensors_to_non_fp8", force_all_tensors_to_non_fp8_patched)


if os.getenv("FP8", "False") == "True":
    import transformer_engine.common.recipe as te_recipe
    import transformer_engine.pytorch as te
    from megatron.bridge.models.gpt_provider import GPTModelProvider

    _original_provide = GPTModelProvider.provide

    def provide_with_fp8(self, pre_process=None, post_process=None, vp_stage=None):
        recipe = te_recipe.DelayedScaling()
        with te.fp8_model_init(recipe=recipe):
            return _original_provide(self, pre_process=pre_process, post_process=post_process, vp_stage=vp_stage)

    GPTModelProvider.provide = provide_with_fp8


def resolve(cfg):
    OmegaConf.register_new_resolver("add", lambda x, y: x + y)
    OmegaConf.register_new_resolver("multiply", lambda x, y: x * y)
    OmegaConf.register_new_resolver("floor_div", lambda x, y: x // y)
    OmegaConf.register_new_resolver("ceil_div", lambda x, y: ceil(x / y))
    OmegaConf.register_new_resolver("floor", lambda x: floor(x))
    OmegaConf.resolve(cfg)
    return cfg


def get_rank():
    return int(os.getenv("SLURM_PROCID", 0))


@property
def pack_metadata_override(self):
    return None

from megatron.bridge.data.builders import FinetuningDatasetBuilder
FinetuningDatasetBuilder.pack_metadata = pack_metadata_override


def patch_build_train_valid_test_data_loaders(cfg, train_state, build_train_valid_test_datasets_provider, dp_group, *, eval_dp_group):
    from megatron.bridge.data.loaders import build_train_valid_test_datasets
    from megatron.bridge.data.samplers import build_pretraining_data_loader
    from megatron.bridge.training.utils.sig_utils import DistributedSignalHandler
    from megatron.core import mpu

    # Construct the data pipeline
    # Build datasets.
    train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
        cfg=cfg,
        build_train_valid_test_datasets_provider=build_train_valid_test_datasets_provider,
    )

    exit_signal = cfg.train.exit_signal

    def worker_init_fn(_):
        DistributedSignalHandler(exit_signal).__enter__()

    maybe_worker_init_fn = worker_init_fn if cfg.train.exit_signal_handler_for_dataloader else None

    dp_rank = torch.distributed.get_rank(group=dp_group)
    dp_size = torch.distributed.get_world_size(group=dp_group)

    # Build dataloders.
    train_dataloader = build_pretraining_data_loader(
        train_ds,
        train_state.consumed_train_samples,
        cfg.dataset.dataloader_type,
        cfg.train.micro_batch_size,
        cfg.dataset.num_workers,
        cfg.dataset.data_sharding,
        worker_init_fn=maybe_worker_init_fn,
        collate_fn=train_ds.collate_fn if hasattr(train_ds, "collate_fn") else None,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        global_batch_size=cfg.train.global_batch_size,
    )

    valid_dataloader = build_pretraining_data_loader(
        valid_ds,
        train_state.consumed_valid_samples,
        "batch",
        cfg.validation.eval_micro_batch_size,
        cfg.dataset.num_workers,
        cfg.dataset.data_sharding,
        worker_init_fn=maybe_worker_init_fn,
        collate_fn=valid_ds.collate_fn if hasattr(valid_ds, "collate_fn") else None,
        pin_memory=cfg.dataset.pin_memory,
        persistent_workers=cfg.dataset.persistent_workers,
        data_parallel_rank=dp_rank,
        data_parallel_size=dp_size,
        global_batch_size=cfg.validation.eval_global_batch_size,
        drop_last=False,
    )

    train_state.do_train = True
    train_state.do_valid = True
    train_state.do_test = False

    return train_dataloader, valid_dataloader, None


import megatron.bridge.data.loaders as loaders
loaders.build_train_valid_test_data_loaders = patch_build_train_valid_test_data_loaders


def masked_next_token_loss_patch(
    output_tensor,
    loss_mask,
    is_training,
):
    masked_sum = torch.sum(output_tensor * loss_mask, dim=1)
    num_valid_tokens = loss_mask.sum(1)

    if parallel_state.get_context_parallel_world_size() > 1 and not is_training:
        cp_group = parallel_state.get_context_parallel_group()
        work_tokens = torch.distributed.all_reduce(num_valid_tokens, group=cp_group, async_op=True)
        work_sums = torch.distributed.all_reduce(masked_sum, group=cp_group, async_op=True)
        work_tokens.wait()
        work_sums.wait()

    loss = torch.nan_to_num(masked_sum / num_valid_tokens, nan=0.0)

    if not is_training:
        sample_present = (num_valid_tokens > 0).long().detach()
        reporting_loss = torch.stack([loss.detach(), sample_present], dim=1)
        return reporting_loss, {"lm loss": reporting_loss}

    return loss, {"lm loss": loss}


from functools import partial
from megatron.bridge.training.utils.pg_utils import get_pg_collection
from megatron.bridge.training.gpt_step import get_batch

def forward_step(state, data_iterator, model, return_schedule_plan: bool = False):
    pg_collection = get_pg_collection(model)
    tokens, labels, loss_mask, attention_mask, position_ids, _ = get_batch(
        data_iterator, state.cfg, False, pg_collection=pg_collection
    )
    forward_args = {
        "input_ids": tokens,
        "position_ids": position_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
    output_tensor = model(**forward_args)
    loss_function = partial(masked_next_token_loss_patch, loss_mask=loss_mask, is_training=model.training)

    return output_tensor, loss_function


import megatron.bridge.training.losses as losses
losses.masked_next_token_loss = masked_next_token_loss_patch

import megatron.bridge.training.config as config
_original_validate = config.ConfigContainer.validate


def validate_patched(self):
    need_temp_override = self.model.context_parallel_size > 1 and self.model.calculate_per_token_loss is False

    orig = self.model.calculate_per_token_loss
    if need_temp_override:
        self.model.calculate_per_token_loss = True

    try:
        _original_validate(self)
    except AssertionError as e:
        raise
    finally:
        if need_temp_override:
            self.model.calculate_per_token_loss = orig

config.ConfigContainer.validate = validate_patched

from megatron.bridge.training import eval as eval_module
from megatron.bridge.training.utils.pg_utils import get_pg_collection


_orig_evaluate = eval_module.evaluate
# Evaluate only on the data parallel group as CP group is already included in masked_next_token_loss_patch 
def evaluate_dp_only(*args, **kwargs):
    model = kwargs.get("model", args[3] if len(args) > 3 else None)
    pg = kwargs.get("pg_collection") or get_pg_collection(model)
    saved_dp_cp = pg.dp_cp
    pg.dp_cp = pg.dp
    try:
        return _orig_evaluate(*args, **kwargs)
    finally:
        pg.dp_cp = saved_dp_cp

eval_module.evaluate = evaluate_dp_only
