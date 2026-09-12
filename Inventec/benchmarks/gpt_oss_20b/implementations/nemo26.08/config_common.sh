export TRAIN_ONLY=0

export USE_DIST_OPTIMIZER=True

export NVTE_FWD_LAYERNORM_SM_MARGIN=16
export NVTE_BWD_LAYERNORM_SM_MARGIN=16

export NCCL_MIN_NCHANNELS=4

export CUDA_DEVICE_MAX_CONNECTIONS=1

export MICRO_BATCH_SIZE=1

: "${LOAD_MINIMAL_NUM_SAMPLES:=0}"

if [[ "${LOAD_MINIMAL_NUM_SAMPLES}" -eq 1 ]]; then
  export MAX_STEPS=500
  export OVERRIDE_ZERO_CONSUMED_SAMPLES=0
  export INIT_GLOBAL_STEP=0
fi

export TE_UB_ATOMIC_GEMM_RS=0
export MC_TP_OVERLAP_AG=True
export MC_TP_OVERLAP_RS=True

export NCCL_P2P_NET_CHUNKSIZE=2097152

# Enable per-communicator nccl option tuning
export NCCL_CFG_PATH="/workspace/llm/conf/nccl/custom_communicator_cta.yaml"

export NCCL_SHARP_GROUP_SIZE_THRESH=2  # Avoid falling back to non-sharp

export FP8=True
export FP8_HYBRID=True

export NCCL_WORK_FIFO_DEPTH=1048576

export DEFER_EMBEDDING_WGRAD_COMPUTE=True
export WGRAD_DEFERRAL_LIMIT=50

export OVERLAP_GRAD_REDUCE=True
export OVERLAP_PARAM_GATHER=True
export ALIGN_PARAM_GATHER=True
export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=True
export FP8_PARAM_GATHER=True
export FUSED_QKV_ROPE=True

# To silent warnings that print during training
export TOKENIZERS_PARALLELISM=False

export HYDRA_FULL_ERROR=1
export HF_HUB_OFFLINE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TORCH_CPP_LOG_LEVEL=ERROR
export TORCH_NCCL_AVOID_RECORD_STREAMS=1
export TORCH_NCCL_HIGH_PRIORITY=1

# Add Min/max NCCL CTAs for NCCL communicators
export NCCL_MIN_CTAS=16
export NCCL_MAX_CTAS=32

# Enable NCCL-UB for DP
export NCCL_UB_DP=True

export FP8_RECIPE="tensorwise"
export NVTE_DPA_FP8_RECIPE=""
export NVTE_DPA_FP8DS_REDUCE_AMAX=0

export TP_PP_DP_MAPPING=True
export MODEL_SIZE="20b"

export BINDCMD="bindpcie --cpu=node"
export EXTRA_ARGS=""
export PYTHONWARNINGS="ignore::FutureWarning,ignore::UserWarning"

export LOAD_CHECKPOINT=""
export OVERWRITTEN_NUM_LAYERS=24
export VAL_SAMPLES=1024
export MAX_STEPS=1200000
export OPT_LR_DECAY_STEPS=${MAX_STEPS}

export OVERLAP_PARAM_GATHER_WITH_OPTIM_STEP=False
export DEFER_EMBEDDING_WGRAD_COMPUTE=False
export NVTE_NORM_FWD_USE_CUDNN=1
export NVTE_NORM_BWD_USE_CUDNN=1
export ALIGN_PARAM_GATHER=False
unset PYTORCH_CUDA_ALLOC_CONF
export USE_TE_OPS=False
export CE_FUSION_IMPL=te
export BUCKET_SIZE=768000000

export MODEL_TFLOP_PER_SAMPLE=421.59
export LOG_EVERY_N_STEPS=32
export GC_TRAIN=10000
export GC_VALID=10000

# Enable 2D weights and stochastic rounding if using FP4
export NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING=0
export NVTE_NVFP4_DISABLE_2D_QUANTIZATION=0
