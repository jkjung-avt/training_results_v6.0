source $(dirname ${BASH_SOURCE[0]})/config_common.sh

# no munge in the docker container
export PMIX_MCA_psec=^munge

## DL params
# 1024 per rank x 2 nodes x 4 GPUs = global batch 8192, the lowest of the three
# global batch sizes the upstream reference swept for RCPs (8192/16384/32768).
export BATCH_SIZE=1024

# Reference RCP hyperparameters for GBS 8192. The target (post-warmup) LR is the
# ONLY LR the reference allows to be tuned, and it scales linearly with global
# batch: 1e-6 @ 8192, 2e-6 @ 16384, 4e-6 @ 32768. 1e-6 is therefore the RCP
# value for this config, not a placeholder.
export DENSE_LR=0.000001
export SPARSE_LR=0.000001
# Warmup steps and start LR are FIXED (non-tunable) for a submission: the
# reference pins 24000 / 0.0 across all three batch sizes and produced its whole
# 60-run RCP sweep with them. Same as the gin defaults; set explicitly so the
# submission-relevant values are visible in the config rather than inherited.
# NOTE: at GBS 8192 this ramp spans 24000*8192 = 196.6M samples (8.6% of an
# epoch), while the reference seeds converge at 61.9M-80.3M samples -- i.e.
# convergence lands ~31% into the ramp. The warmup IS the schedule up to the
# target here, so shortening it is not a free speedup.
export LR_WARMUP_STEPS=24000
export LR_WARMUP_START_LR=0.0
export GRAD_CLIP_NORM=1.0

# Full reference sweep to the AUC target. EVAL_EVERY_DATA_PCT=0.001 is the
# cadence the RCP sweep used; samples-to-converge is quantized to this grid, so
# a coarser value is not comparable against rcp_logs/gbs_8192.
export START_TS=0
export NUM_TRAIN_TS=299
export EVAL_EVERY_DATA_PCT=0.001
# Skip periodic eval until 2.5% of the epoch is trained. GBS-DEPENDENT: 0.025 is
# the GBS-8192 value from the upstream fit, and it is only valid on the 0.001
# eval grid set above. Resolves to skip_eval_until_step=6992, so the first eval
# runs at step 7000 (eval #25) while the earliest RCP seed crosses AUC 0.75 at
# step 7560 (eval #27) -- 2 evals of margin. Cuts eval from ~29% to ~6.9% of
# end-to-end. Does NOT touch the training trajectory, only measurement cadence.
export SKIP_EVAL_EPOCH_PCT=0.025
# = config_common.sh; restated because it is the convergence criterion this
# config exists to measure.
export AUC_THRESHOLD=0.75

# balance the skewed-table embedding all-to-all (see yambda_5b.gin notes)
export EMB_SHARDING_OVERRIDES="album_id=column_wise,artist_id=column_wise"

## System run parms
export DGXNNODES=1
export DGXNGPU=8
export DGXSYSTEM=$(basename $(readlink -f ${BASH_SOURCE[0]}) | sed 's/^config_//' | sed 's/\.sh$//' )

export WALLTIME_RUNANDTIME=180
# 30 min prolog headroom: cold nodes rsync the dataset to NVMe
export WALLTIME=$((30 + ${NEXP:-1} * ($WALLTIME_RUNANDTIME + 5)))

export MLPERF_SUBMITTER="Inventec"
export MLPERF_SUBMISSION_ORG="Inventec Corporation"
export MLPERF_CLUSTER_NAME="Inventec AI Lab"
export MLPERF_SYSTEM_NAME="P5800G7 8xB300"
export MLPERF_SUBMISSION_PLATFORM="Inventec P5800G7"
export MLPERF_STATUS="research"
