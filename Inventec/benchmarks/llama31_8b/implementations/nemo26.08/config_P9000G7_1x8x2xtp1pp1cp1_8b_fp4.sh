source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp4.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_8b.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_fp8attn.sh

# no munge in the docker container
export PMIX_MCA_psec=^munge

export MINIBS=2
export MICRO_BATCH_SIZE=2
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export INTERLEAVED_PIPELINE=null
export CONTEXT_PARALLEL=1

# Experimental
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1

export LR=0.00044
export WARMUP_STEPS=16
export VAL_CHECK_INTERVAL=768

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=90
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export MLPERF_SUBMITTER="Inventec"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_CLUSTER_NAME="Inventec AI Lab"
export MLPERF_SYSTEM_NAME="P9000G7 8xB300"
export MLPERF_SUBMISSION_PLATFORM="Inventec P9000G7"
export MLPERF_STATUS="research"
