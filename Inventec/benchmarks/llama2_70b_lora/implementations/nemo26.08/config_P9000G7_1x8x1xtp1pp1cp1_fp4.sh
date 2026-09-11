#!/bin/bash

source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh

# no munge in the docker container
export PMIX_MCA_psec=^munge

# hyperparameters
export MAX_STEPS=450
export LR=0.00075
export MINIBS=1
export CP=1
export MCORE_CUDA_GRAPH=1
export BUCKET_SIZE=10000000
export NUM_WORKERS=8

export HEALING_ITER=350

# system parameters
export VBOOST_VALUE=0
export DGXNNODES=1
export DGXNGPU=8
export WALLTIME_RUNANDTIME=18
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 2)))

export MLPERF_SUBMITTER="Inventec"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_CLUSTER_NAME="Inventec AI Lab"
export MLPERF_SYSTEM_NAME="P9000G7 8xB300"
export MLPERF_SUBMISSION_PLATFORM="Inventec P9000G7"
export MLPERF_STATUS="research"
