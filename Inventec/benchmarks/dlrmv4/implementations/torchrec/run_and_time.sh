#!/bin/bash

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

###########################################################################
# Runs inside the container, one copy per GPU rank. slurm2pytorch supplies the
# standard torch distributed environment from the corresponding Slurm task.
#
# This script MUST NOT have any SLURM dependences (no use of SLURM envvars).
# It expects the torchrun-style vars (RANK, LOCAL_RANK, WORLD_SIZE,
# LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT) from slurm2pytorch.
###########################################################################

# Vars supplied by slurm2pytorch.
: "${RANK:?RANK not set}"
: "${LOCAL_RANK:?LOCAL_RANK not set}"
: "${WORLD_SIZE:?WORLD_SIZE not set}"
: "${LOCAL_WORLD_SIZE:?LOCAL_WORLD_SIZE not set}"
: "${MASTER_ADDR:?MASTER_ADDR not set}"
: "${MASTER_PORT:?MASTER_PORT not set}"

# no `set -e`: the training exit code is propagated explicitly
set -x


: "${DATASET:=yambda-5b}"
: "${TRAIN_MODE:=streaming-train-eval}"
export DLRM_DATA_PATH="${DLRM_DATA_PATH:-/data}"

# fallbacks for a run without config_common.sh; a config-provided value wins
export AUC_THRESHOLD="${AUC_THRESHOLD:-0.75}"
export CKPT_TIME_INTERVAL_S="${CKPT_TIME_INTERVAL_S:-0}"
export MLPERF_LOGGING="${MLPERF_LOGGING:-1}"

if [ "${RANK}" -eq 0 ]; then
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
fi

: "${LOGGER:=""}"
if [[ -n "${APILOG_DIR:-}" ]]; then
    if [ "${RANK}" -eq 0 ]; then
      LOGGER="apiLog.sh -p MLPerf/${MODEL_NAME} -v ${FRAMEWORK}/train/${DGXSYSTEM}"
    fi
fi

${LOGGER:-} ${BINDCMD:-} python -u -m generative_recommenders.dlrm_v4.train.train_ranker \
    --dataset "${DATASET}" \
    --mode "${TRAIN_MODE}"; ret_code=$?

if [[ $ret_code != 0 ]]; then exit $ret_code; fi

if [ "${RANK}" -eq 0 ]; then
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo "ENDING TIMING RUN AT $end_fmt"
    result=$(( $end - $start ))
    echo "RESULT,${MODEL_NAME},,$result,nvidia,$start_fmt"
fi
