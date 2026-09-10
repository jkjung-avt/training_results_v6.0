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


import gc

import megatron.core.fp4_utils
import torch
import transformer_engine.pytorch as te
import transformer_engine_torch as tex
from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.bridge.utils.common_utils import print_rank_0
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from transformer_engine.common.recipe import (
    DelayedScaling,
    MXFP8BlockScaling,
    NVFP4BlockScaling,
)
from transformer_engine.pytorch.tensor.float8_tensor import Float8Quantizer
from transformer_engine.pytorch.tensor.mxfp8_tensor import MXFP8Quantizer
from transformer_engine.pytorch.tensor.nvfp4_tensor import NVFP4Quantizer


def extract_module(module):
    current = module
    while True:
        if isinstance(current, (list, tuple)):
            current = current[0]
            continue
        child = getattr(current, "module", None)
        if child is None:
            return current
        current = child


class NVFP4Callback(Callback):
    def __init__(self, cfg):
        self.cfg = cfg
        # Healing states
        self.healing_precision = cfg.model.healing_precision
        self.healing_lambda = None
        self.healing_iter = cfg.model.healing_iter

        # Pre-quantization
        self.pre_quantized_model = cfg.model.pre_quantized_model
        self.nvfp4_quantizer = None
        self.fp4_cpu_params = []
        self.fp8_quantizer = None
        self.fp8_cpu_params = []
        self.store_quantized_params_on_gpu = cfg.model.store_gpu

        # BF16 layers
        self.first_last_layers_bf16 = cfg.model.first_last_layers_bf16
        self.num_layers_at_start_in_bf16 = cfg.model.num_layers_at_start_in_bf16
        self.num_layers_at_end_in_bf16 = cfg.model.num_layers_at_end_in_bf16

        self.fp8_dpa = cfg.model.fp8_dot_product_attention
        self.fp8_amax_history_len = cfg.model.fp8_amax_history_len
        self.fp8_amax_compute_algo = cfg.model.fp8_amax_compute_algo
        self.reduce_amax = cfg.model.reduce_amax

    def _healing_setup(self) -> None:
        def fp4_recipe(_config):
            return NVFP4BlockScaling(fp8_dpa=self.fp8_dpa)

        megatron.core.fp4_utils.get_fp4_recipe = fp4_recipe

        if self.healing_precision == "FP8_DS":
            self.healing_lambda = lambda config: DelayedScaling(
                amax_history_len=self.fp8_amax_history_len,
                amax_compute_algo=self.fp8_amax_compute_algo,
                reduce_amax=self.reduce_amax,
                fp8_dpa=self.fp8_dpa,
            )
        elif self.healing_precision == "MXFP8":
            self.healing_lambda = lambda config: MXFP8BlockScaling()
        else:
            raise ValueError(f"Unsupported healing precision: {self.healing_precision}")

    def _pre_quantize_model(self, model):
        """Pre-quantize model to FP8 and NVFP4"""
        self.nvfp4_quantizer = NVFP4Quantizer()

        # Set up FP8 quantizer for healing, which can be in either FP8 DS or MXFP8
        if self.healing_precision == "FP8_DS":
            self.fp8_quantizer = Float8Quantizer(
                scale=torch.ones(1, dtype=torch.float32, device=torch.cuda.current_device()),
                amax=torch.zeros(1, dtype=torch.float32, device=torch.cuda.current_device()),
                fp8_dtype=tex.DType.kFloat8E4M3,
            )
        elif self.healing_precision == "MXFP8":
            self.fp8_quantizer = MXFP8Quantizer(tex.DType.kFloat8E4M3)
        else:
            raise ValueError(f"Unsupported healing precision: {self.healing_precision}")

        self.fp8_cpu_params = self._get_quantized_params_cpu(model, self.fp8_quantizer, self.healing_precision)
        _ = self._get_quantized_params_cpu(model, self.nvfp4_quantizer, "NVFP4", replace=True)

        # We now have NVFP4 parameters in GPU memory (for training)
        # and FP8 parameters in host memory (for healing)
        self.pre_quantized_model = True

    def _get_quantized_params_cpu(self, model, quantizer, qtype: str, replace: bool = False):
        # Unwrap layers from model
        extracted_module = extract_module(model)
        layers = extracted_module.decoder.layers
        layer_count = len(layers)

        quantized_params = []
        for layer_idx, layer in enumerate(layers):
            # Skip first and last BF16 layers
            if self.first_last_layers_bf16:
                if (
                    layer_idx < self.num_layers_at_start_in_bf16
                    or layer_idx >= layer_count - self.num_layers_at_end_in_bf16
                ):
                    quantized_params.append([])  # Append empty list for consistency
                    continue

            # Quantize weights of TE modules
            quantized_layer_params = []
            for name, module in layer.named_modules():
                if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                    continue
                if not hasattr(module, "weight"):
                    continue
                param = module.weight
                with torch.no_grad():
                    qparam = quantizer(param.detach())
                    # Validate quantized tensor based on type
                    if qtype in ["MXFP8", "NVFP4"]:
                        assert qparam._rowwise_data is not None, "No rowwise data."
                        assert qparam._columnwise_data is not None, "No columnwise data."
                    elif qtype == "FP8_DS":
                        assert qparam._data is not None, "No data."
                        # Float8Tensor may or may not have transpose data
                    else:
                        raise ValueError(f"Unsupported quantization type: {qtype}")

                    if replace:
                        setattr(
                            module,
                            "weight",
                            torch.nn.Parameter(qparam, requires_grad=False),
                        )
                    else:
                        qparam = qparam.clone()
                        if not self.store_quantized_params_on_gpu:
                            # Move data to CPU with pinned memory for faster H2D transfer later
                            if qtype in ["MXFP8", "NVFP4"]:
                                qparam._rowwise_data = qparam._rowwise_data.cpu().pin_memory()
                                qparam._columnwise_data = qparam._columnwise_data.cpu().pin_memory()
                            elif qtype == "FP8_DS":
                                qparam._data = qparam._data.cpu().pin_memory()
                                if hasattr(qparam, "_transpose") and qparam._transpose is not None:
                                    qparam._transpose = qparam._transpose.cpu().pin_memory()
                        quantized_layer_params.append(qparam)
            quantized_params.append(quantized_layer_params)

        return quantized_params

    def _set_quantized_params_cpu(self, model, cpu_params, qtype: str):
        # Unwrap layers from model
        extracted_module = extract_module(model)
        layers = extracted_module.decoder.layers
        layer_count = len(layers)
        device = torch.cuda.current_device()

        # Use a dedicated stream for H2D transfers
        transfer_stream = torch.cuda.Stream()

        # Collect (module, old_weight, new_weight) tuples for batch processing
        weight_swaps = []

        with torch.no_grad():
            # ============ Transfer ALL weights to GPU (fully async) ============
            with torch.cuda.stream(transfer_stream):
                for layer_idx, layer in enumerate(layers):
                    if self.first_last_layers_bf16:
                        if (
                            layer_idx < self.num_layers_at_start_in_bf16
                            or layer_idx >= layer_count - self.num_layers_at_end_in_bf16
                        ):
                            continue

                    for name, module in layer.named_modules():
                        if not isinstance(module, (te.Linear, te.LayerNormLinear)):
                            continue
                        if not hasattr(module, "weight"):
                            continue

                        weight = cpu_params[layer_idx].pop(0)
                        old_weight = module.weight

                        if not self.store_quantized_params_on_gpu:
                            # Fully async H2D transfer - no sync, no waiting
                            if qtype in ["MXFP8", "NVFP4"]:
                                weight._rowwise_data = weight._rowwise_data.to(device, non_blocking=True)
                                weight._columnwise_data = weight._columnwise_data.to(device, non_blocking=True)
                            elif qtype == "FP8_DS":
                                weight._data = weight._data.to(device, non_blocking=True)
                                if hasattr(weight, "_transpose") and weight._transpose is not None:
                                    weight._transpose = weight._transpose.to(device, non_blocking=True)
                            else:
                                raise ValueError(f"Unsupported quantization type: {qtype}")

                        # Store for batch replacement later
                        weight_swaps.append((module, old_weight, weight))

            # Wait for ALL transfers to complete
            transfer_stream.synchronize()

            # ============ Swap all weights (instant - just pointer assignments) ============
            layer_modules = {}  # Track which layers need fuser update
            for module, old_weight, new_weight in weight_swaps:
                setattr(module, "weight", torch.nn.Parameter(new_weight, requires_grad=False))
                # Find parent layer for fuser updates
                for layer_idx, layer in enumerate(layers):
                    for name, mod in layer.named_modules():
                        if mod is module:
                            layer_modules[layer_idx] = layer
                            break

            # ============ Clear old weights ============
            for module, old_weight, new_weight in weight_swaps:
                if hasattr(old_weight, "clear"):
                    old_weight.clear()

            if self.cfg.model.use_transformer_engine_op_fuser:
                for layer_idx, layer in layer_modules.items():
                    layer.mlp._fused_impl = (layer.mlp._make_fused_impl(),)
                    layer.self_attention.linear_proj._fused_branches = (
                        layer.self_attention.linear_proj._make_fused_branches()
                    )
                    layer.self_attention.linear_qkv._fused_branches = (
                        layer.self_attention.linear_qkv._make_fused_branches()
                    )

    def _reset_cuda_graphs(self):
        torch.distributed.barrier()
        if not hasattr(FullCudaGraphWrapper, "cuda_graph"):
            return

        FullCudaGraphWrapper.cuda_graph["training"] = None
        FullCudaGraphWrapper.cuda_graph["validation"] = None
        FullCudaGraphWrapper.result["training"] = None
        FullCudaGraphWrapper.result["validation"] = None

        if getattr(self.cfg.model, "reset_cg_after_healing", False):
            FullCudaGraphWrapper.curr_iteration["training"] = 0
            FullCudaGraphWrapper.curr_iteration["validation"] = 0

    def on_data_init_start(self, context: CallbackContext):
        if self.pre_quantized_model:
            gc.collect()
            torch.cuda.empty_cache()
            self._healing_setup()
            self._pre_quantize_model(context.model)

    def on_train_step_end(self, context: CallbackContext) -> None:
        if context.state.train_state.step + 1 != self.healing_iter:
            return

        print_rank_0("FP8 Healing starting...")
        self._reset_cuda_graphs()

        # Switch to pre-quantized FP8 parameters
        if self.pre_quantized_model:
            assert self.fp8_cpu_params, "FP8 parameters not found"
            self._set_quantized_params_cpu(context.model, self.fp8_cpu_params, self.healing_precision)

        megatron.core.fp4_utils.get_fp4_recipe = self.healing_lambda
