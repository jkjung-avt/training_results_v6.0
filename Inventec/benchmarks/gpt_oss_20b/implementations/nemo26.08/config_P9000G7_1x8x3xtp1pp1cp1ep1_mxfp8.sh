source $(dirname ${BASH_SOURCE[0]})/config_common.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_mxfp8.sh
source $(dirname ${BASH_SOURCE[0]})/config_common_cg.sh

# no munge in the docker container
export PMIX_MCA_psec=^munge

export MINIBS=3
export MICRO_BATCH_SIZE=3
export TENSOR_MODEL_PARALLEL=1
export SEQ_PARALLEL=False
export PIPELINE_MODEL_PARALLEL=1
export CONTEXT_PARALLEL=1
export EXPERT_PARALLEL=1
# HybridEP config
export NUM_OF_HYBRID_EP_RANKS_PER_NVLINK_DOMAIN=$EXPERT_PARALLEL
export USE_MNNVL=0

# 1.5x oversized activation for CG. Override the env. var set in config_common_cg.sh
export MOE_EXPERT_RANK_CAPACITY_FACTOR=1.5

# Enable CuteDSL kernels
export USE_TE_OPS=True
export NVTE_CUTEDSL_FUSED_GROUPED_MLP=1
unset CUDNN_FE_GROUPED_GEMM_DYNAMIC_MNKL

export LR=0.0005
export VAL_CHECK_INTERVAL=512
export LR_WARMUP_STEPS=256

export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=100
export WALLTIME=$((5 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export MLPERF_SUBMITTER="Inventec"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_CLUSTER_NAME="Inventec AI Lab"
export MLPERF_SYSTEM_NAME="P9000G7 8xB300"
export MLPERF_SUBMISSION_PLATFORM="Inventec P9000G7"
export MLPERF_STATUS="research"
