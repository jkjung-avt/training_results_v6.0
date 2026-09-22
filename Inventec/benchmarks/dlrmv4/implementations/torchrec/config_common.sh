# Hyperparameters map to the gin env bindings in
# generative_recommenders/dlrm_v4/train/gin/yambda_5b.gin.

export DATASET="yambda-5b"
export TRAIN_MODE="streaming-train-eval"

# must match the staged hstu_cache_L2039 or the first run rebuilds it
export HISTORY_LENGTH=4086
export MIN_HISTORY=4086
export MAX_SEQ_LEN=4096

# bf16 is only numerically safe with the fused TRITON kernels
export HSTU_HAMMER_KERNEL=TRITON
# Use the SM10x cuDNN Frontend HSTU attention implementation while
# retaining the existing Triton preprocessing, normalization, and output path.
export HSTU_USE_CUDNN_FE="${HSTU_USE_CUDNN_FE:-1}"
# the pinned autotune configs are MI350X-tuned
export TRITON_FULL_AUTOTUNE=1

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Constrain every Yambda embedding table to row-wise sharding by default.
export EMB_SHARDING_OVERRIDES="${EMB_SHARDING_OVERRIDES:-item_id=row_wise,\
artist_id=row_wise,album_id=row_wise,uid=row_wise,user_x_artist=row_wise,\
user_x_album=row_wise,user_x_hour=row_wise,item_x_hour=row_wise,\
artist_x_hour=row_wise,user_x_is_organic=row_wise,\
user_x_artist_x_hour=row_wise}"

# Use torch.gather instead of index_select to restore deduplicated embedding
# outputs. The same knob enables the TorchRec index-dedup path it depends on.
export TORCHREC_USE_GATHER_SELECT="${TORCHREC_USE_GATHER_SELECT:-0}"

# run.sub launches one task per GPU, so bindpcie applies independently to each
# rank and places its worker threads near the assigned GPU.
export BINDCMD="${BINDCMD:-bindpcie --cpu=node}"

# reference convergence target (upstream README target: 0.80275; reference default: 0.75)
export AUC_THRESHOLD=0.75
export MLPERF_LOGGING=1

# lowest_numerical_precision_in_* disclosure (training_6.1.0/common.yaml).
# linear/attn: make_model.bf16_training=True autocasts the whole HSTU
# transducer to bfloat16; dlrmv4 has no fp8/fp4 path.
# comm: the sparse embedding all-to-all is quantized via TorchRec QCommsConfig.
# MUST stay in sync with SPARSE_A2A_FWD/SPARSE_A2A_BWD (gin default fp16/fp16);
# set both of those to fp32 to disable quantization and this becomes fp32.
# Dense gradients are not compressed (no DDP comm hook), so they go out at fp32.
export MLPERF_LINEAR_PRECISION="bfloat16"
export MLPERF_ATTN_PRECISION="bfloat16"
export MLPERF_COMM_PRECISION="fp16"
# cadence (train steps) of the MLPerf tracked_stats train_step_time event
export MLPERF_STEP_TIME_EVERY=10

# CKPT_PATH is unset, so the hourly checkpoint clock would only add a per-step
# NCCL broadcast + host sync; literal 0 disables it ("" falls back to 3600)
export CKPT_TIME_INTERVAL_S=0

export VERIFY_MOUNTS=0
