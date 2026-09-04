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


import time

from megatron.bridge.training.callbacks import Callback, CallbackContext
from megatron.core.full_cuda_graph import FullCudaGraphWrapper
from mlperf_common.frameworks.pyt import PyTCommunicationHandler
from mlperf_common.logging import MLLoggerWrapper


class DeltaTimer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time = time.perf_counter()
        return self.start_time

    def get_delta(self):
        prev_time = self.start_time
        return self.reset() - prev_time


mllogger = MLLoggerWrapper(PyTCommunicationHandler())


class MLPerfLoggingCallback(Callback):
    """MLPerf logging callback."""

    def __init__(self, cfg):
        self.cfg = cfg
        self.global_batch_size = self.cfg.model.global_batch_size
        self.train_block_started = True
        self.train_current_block = 0
        self.force_success = cfg.custom.force_success_status
        self.previous_step = 0
        self.first_eval = True

    def on_data_init_start(self, context: CallbackContext):
        mllogger.log_init_stop_run_start()

    def on_train_start(self, context: CallbackContext):
        context.state.should_stop = False
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: context.state.cfg.validation.start_eval_at_iter * self.global_batch_size,
                "step": context.state.train_state.step,
            },
        )
        self.timer = DeltaTimer()

    def on_train_end(self, context: CallbackContext):
        if self.train_block_started:
            self._end_train_block(context.state)

        FullCudaGraphWrapper.cuda_graph = None

    def on_eval_start(self, context: CallbackContext):
        """Log validation start."""
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        self._log_train_step_time(context.state)
        if self.train_block_started:
            self._end_train_block(context.state)

        mllogger.start(
            mllogger.constants.EVAL_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: self._get_samples_count(context.state),
                "step": self._get_step(context.state),
            },
        )

    def on_eval_end(self, context: CallbackContext):
        if hasattr(context.state, "warmup") and context.state.warmup:
            return
        self._log_custom_timedelta("validation_time", self._get_step(context.state))

        samples_count = self._get_samples_count(context.state)
        loss = context.total_loss_dict["lm loss"].item()

        mllogger.event(
            key=mllogger.constants.EVAL_ACCURACY,
            metadata={mllogger.constants.SAMPLES_COUNT: samples_count},
            value=loss,
        )
        mllogger.end(
            mllogger.constants.EVAL_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: samples_count,
                "step": self._get_step(context.state),
            },
        )

        if loss < self.cfg.custom.target_log_ppl:
            context.state.should_stop = True
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": "success"},
            )
        elif context.state.train_state.step >= self.cfg.trainer.max_steps:
            context.state.should_stop = True
            status = "success" if self.force_success else "aborted"
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": status},
            )

        if not context.state.should_stop:
            self._start_train_block(context.state)
        else:
            context.state.train_state.step = self.cfg.trainer.max_steps + 1
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def on_train_step_end(self, context: CallbackContext):
        step = context.state.train_state.step + 1
        last_step = step >= self.cfg.trainer.max_steps
        eval_after_this_step = step % context.state.cfg.validation.eval_interval == 0
        if last_step and not eval_after_this_step:
            samples_count = self._get_samples_count(context.state)
            status = "success" if self.force_success else "aborted"
            self._end_train_block(context.state)
            mllogger.end(
                mllogger.constants.RUN_STOP,
                metadata={mllogger.constants.SAMPLES_COUNT: samples_count, "status": status},
            )
            self.train_block_started = False
            context.state.should_stop = True
            context.state.train_state.do_valid = False
            context.state.train_state.do_test = False

    def _start_train_block(self, global_state) -> None:
        self.train_block_started = True
        mllogger.start(
            mllogger.constants.BLOCK_START,
            metadata={
                mllogger.constants.SAMPLES_COUNT: global_state.cfg.validation.eval_interval * self.global_batch_size,
                "step": self._get_step(global_state),
            },
        )

    def _end_train_block(self, global_state) -> None:
        if self.first_eval:
            samples_count = global_state.cfg.validation.start_eval_at_iter * self.global_batch_size
            self.first_eval = False
        else:
            samples_count = global_state.cfg.validation.eval_interval * self.global_batch_size

        mllogger.end(
            mllogger.constants.BLOCK_STOP,
            metadata={
                mllogger.constants.SAMPLES_COUNT: samples_count,
                "step": self._get_step(global_state),
            },
        )
        self.train_block_started = False

    def _log_train_step_time(self, global_state) -> None:
        delta_t = self.timer.get_delta()
        global_step = self._get_step(global_state)
        delta_step = global_step - self.previous_step
        mllogger.event(
            key="tracked_stats",
            metadata={mllogger.constants.SAMPLES_COUNT: delta_step * self.global_batch_size},
            value={
                "train_step_time": delta_t / (delta_step + 1e-8),
            },
        )

        self.previous_step = global_step

    def _log_custom_timedelta(self, value_key, step: int = 0):
        mllogger.event(
            key="tracked_stats",
            metadata={"step": step},
            value={value_key: self.timer.get_delta()},
        )

    def _get_step(self, global_state):
        return global_state.train_state.step

    def _get_samples_count(self, global_state):
        return self._get_step(global_state) * self.global_batch_size


class DeltaTimingCallback(Callback):
    def __init__(self, cfg):
        self.t0 = 0
        self.total_train_step_time = [0, 0]
        self.global_batch_size = cfg.model.global_batch_size
        self.log_every_n_steps = cfg.trainer.log_every_n_steps

    def on_train_start(self, context: CallbackContext):
        self.t0 = time.time()

    def on_train_step_end(self, context: CallbackContext):
        t1 = time.time()
        d = t1 - self.t0
        self.total_train_step_time[0] += d
        self.total_train_step_time[1] += 1
        self.t0 = t1

        step = context.state.train_state.step + 1
        if step % self.log_every_n_steps == 0:
            mllogger.event(
                key="tracked_stats",
                metadata={mllogger.constants.SAMPLES_COUNT: self.global_batch_size * step},
                value={
                    "train_step_time": d,
                    "reduced_train_loss": context.loss_dict["lm loss"].item(),
                },
            )

    def on_eval_end(self, context: CallbackContext):
        """Reset timer after validation to avoid including validation time in first train step."""
        self.t0 = time.time()
