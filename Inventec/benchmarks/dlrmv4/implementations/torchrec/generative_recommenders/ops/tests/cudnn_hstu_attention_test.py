# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

#!/usr/bin/env python3

# pyre-strict

import os
import unittest
from contextlib import ExitStack
from unittest import mock

import torch
import generative_recommenders.ops.cudnn_hstu_attention as cudnn_hstu_attention
from generative_recommenders.ops.cudnn_hstu_attention import (
    _prepare_cudnn_hstu_metadata,
    cudnn_hstu_attention_bwd,
    use_cudnn_hstu_attention,
)


class CuDNNHSTUAttentionTest(unittest.TestCase):
    def test_switch_defaults_to_enabled(self) -> None:
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertTrue(use_cudnn_hstu_attention())
        for value in ("1", "true", "YES", "on"):
            with mock.patch.dict(
                os.environ, {"HSTU_USE_CUDNN_FE": value}, clear=True
            ):
                self.assertTrue(use_cudnn_hstu_attention())
        for value in ("0", "false", "no", "off"):
            with mock.patch.dict(
                os.environ, {"HSTU_USE_CUDNN_FE": value}, clear=True
            ):
                self.assertFalse(use_cudnn_hstu_attention())

    def test_plain_causal_and_local_masks_use_window_metadata(self) -> None:
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        cu, window = _prepare_cudnn_hstu_metadata(
            seq_offsets=offsets,
            num_targets=None,
            max_attn_len=0,
            contextual_seq_len=0,
        )
        self.assertEqual(cu.dtype, torch.int32)
        self.assertEqual(window, (-1, 0))

        _, window = _prepare_cudnn_hstu_metadata(
            seq_offsets=offsets,
            num_targets=None,
            max_attn_len=2,
            contextual_seq_len=0,
        )
        self.assertEqual(window, (2, 0))

    def test_target_aware_and_contextual_masks_are_rejected(self) -> None:
        offsets = torch.tensor([0, 4], dtype=torch.int64)
        with self.assertRaisesRegex(NotImplementedError, "plain causal"):
            _prepare_cudnn_hstu_metadata(
                seq_offsets=offsets,
                num_targets=torch.ones(1, dtype=torch.int64),
                max_attn_len=0,
                contextual_seq_len=0,
            )
        with self.assertRaisesRegex(NotImplementedError, "plain causal"):
            _prepare_cudnn_hstu_metadata(
                seq_offsets=offsets,
                num_targets=None,
                max_attn_len=0,
                contextual_seq_len=1,
            )

    def test_backward_passes_strided_gradients_directly_to_cudnn(self) -> None:
        shape = (4, 2, 64)
        dout = torch.empty(shape, dtype=torch.bfloat16)
        q = torch.empty(shape, dtype=torch.bfloat16)
        k = torch.empty(shape, dtype=torch.bfloat16)
        v = torch.empty(shape, dtype=torch.bfloat16)
        grad_storage = torch.empty((4, 8 * 64), dtype=torch.bfloat16)
        _, dv_2d, dq_2d, dk_2d = grad_storage.split(2 * 64, dim=1)
        dv, dq, dk = (
            grad.view(shape) for grad in (dv_2d, dq_2d, dk_2d)
        )
        for grad in (dq, dk, dv):
            self.assertFalse(grad.is_contiguous())
            self.assertEqual(grad.stride(), (512, 64, 1))

        backward = mock.Mock()
        stream = object()

        with ExitStack() as stack:
            stack.enter_context(
                mock.patch.object(
                    cudnn_hstu_attention,
                    "_load_cudnn_hstu_apis",
                    return_value=(mock.Mock(), backward),
                )
            )
            empty_like = stack.enter_context(mock.patch.object(torch, "empty_like"))
            copy_ = stack.enter_context(mock.patch.object(torch.Tensor, "copy_"))
            stack.enter_context(
                mock.patch.object(
                    torch.cuda,
                    "current_stream",
                    return_value=stream,
                )
            )
            cudnn_hstu_attention_bwd(
                dout=dout,
                q=q,
                k=k,
                v=v,
                dq=dq,
                dk=dk,
                dv=dv,
                seq_offsets=torch.tensor([0, 4], dtype=torch.int64),
                num_targets=None,
                N=4,
                alpha=1.0,
                max_attn_len=0,
                contextual_seq_len=0,
            )

        empty_like.assert_not_called()
        copy_.assert_not_called()
        backward.assert_called_once()
        backward_args = backward.call_args.kwargs
        self.assertIs(backward_args["dq_tensor"], dq)
        self.assertIs(backward_args["dk_tensor"], dk)
        self.assertIs(backward_args["dv_tensor"], dv)
        self.assertIs(backward_args["stream"], stream)


if __name__ == "__main__":
    unittest.main()
