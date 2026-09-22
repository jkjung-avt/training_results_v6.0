## Running NVIDIA Generative Recommender DLRMv4 (HSTU) PyTorch MLPerf Benchmark

This file contains the instructions for running the NVIDIA Generative
Recommender DLRMv4 (HSTU) PyTorch MLPerf Benchmark on NVIDIA hardware.

Vendored from the MLPerf Training reference
[mlcommons/training](https://github.com/mlcommons/training/tree/master/recommendation),
originally added via [mlcommons/training#889](https://github.com/mlcommons/training/pull/889).

## 1. Hardware Requirements

- At least 300GB disk space is required for the preprocessed dataset, plus
  another 200GB for the HuggingFace download/conversion cache (`HF_HOME`,
  see below) unless it shares the same volume.
- NVIDIA GPU with at least 288GB memory is strongly recommended.
- GPUs are not required for dataset preparation.

## 2. Software Requirements

- Slurm with [Pyxis](https://github.com/NVIDIA/pyxis) and [Enroot](https://github.com/NVIDIA/enroot)
- [Docker](https://www.docker.com/)

## 3. Set up

### 3.1 Build the container

Replace `<docker/registry>` with your container registry and build:

```bash
docker build -t <docker/registry>/mlperf-nvidia:dlrmv4-pyt .
# optionally: docker push <docker/registry>/mlperf-nvidia:dlrmv4-pyt
export CONT=<docker/registry>/mlperf-nvidia:dlrmv4-pyt
```

make sure that container is accessible on your Slurm system.

### 3.2 Prepare dataset

Set the directory for the data to be downloaded to:

```bash
export DLRM_DATA_PATH=<path/to/dataset>
```

Also redirect the HuggingFace cache off your home directory — the
raw-to-parquet conversion step can use 200GB+ there, which easily exceeds a
typical home quota:

```bash
export HF_HOME=<path/to/hf_cache>
mkdir -p ${DLRM_DATA_PATH} ${HF_HOME}
```

Download and preprocess Yambda-5b (from HuggingFace). The scripts need the
container's Python environment, so run them inside the container with both
directories mounted, using `docker run` or `srun`:

```bash
# docker
docker run --rm --network=host --ipc=host --volume ${DLRM_DATA_PATH}:${DLRM_DATA_PATH} --volume ${HF_HOME}:${HF_HOME} -e DLRM_DATA_PATH -e HF_HOME $CONT bash -c './download_dataset.sh && ./verify_dataset.sh'
```

```bash
# slurm
srun --nodes=1 -t <time> --container-image=${CONT} --container-mounts=${DLRM_DATA_PATH}:${DLRM_DATA_PATH},${HF_HOME}:${HF_HOME} --container-workdir=/workspace/dlrmv4 -p <partition> -A <account> bash -c './download_dataset.sh && ./verify_dataset.sh'
```

At the end, the directory structure should look like:

```
raw/5b/multi_event.parquet
shared_metadata/{artist_item_mapping,album_item_mapping,embeddings}.parquet
processed_5b/{train_sessions,test_events,session_index}.parquet
processed_5b/item_popularity.npy
processed_5b/split_meta.json
```

The first training run additionally builds an `hstu_cache_L<HISTORY_LENGTH>/`
mmap cache under `processed_5b/`, so the data mount must be writable.

### 3.3 Model and checkpoint preparation

#### 3.3.1 Publication/Attribution

The model is **HSTU** (Hierarchical Sequential Transduction Units), the
generative-recommender architecture from Meta's ICML'24 paper *Actions Speak
Louder than Words: Trillion-Parameter Sequential Transducers for Generative
Recommendations* ([arXiv:2402.17152](https://arxiv.org/abs/2402.17152)).

#### 3.3.2 Model Architecture

HSTU replaces the feature-interaction stack of a classic DLRM with a stack of
pointwise-attention "transducer" layers operating over the user's interaction
sequence: sparse embedding tables (sharded with TorchRec
`DistributedModelParallel`) feed an HSTU attention stack, computed with a
fused jagged-attention kernel in bf16, trained against a single `listen_plus`
binary task.

#### 3.3.3 Model checkpoint

DLRMv4 is trained from scratch and does not use a starting checkpoint.

## 4. Launch training

```bash
export LOGDIR=</path/to/output/dir>
export DATADIR=<as/set/above>
export CONT=<as/set/above>
source config_GB300_2x4x1024.sh
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

Set `HSTU_USE_CUDNN_FE=0` to fall back to the Triton attention path instead of
the default cuDNN Frontend implementation.

# 5. Quality

### Quality metric

`window_auc`: AUC on the held-out future evaluation window for the
`listen_plus` task.

### Quality target

0.75

### Evaluation frequency

Evaluate every 0.1% of the training stream (`EVAL_EVERY_DATA_PCT=0.001`).

### Evaluation thoroughness

Every evaluation scores the entire held-out future window, with no
subsampling.

# 6. Additional notes

### Config naming convention

Configuration files follow the format
`config_<SYSTEM>_<NODES>x<GPUS/NODE>x<BATCH/GPU>.sh`. Unlike the
Megatron-based LLM benchmarks, dlrmv4 shards purely with TorchRec
`DistributedModelParallel` (no tensor/pipeline/context parallel), so there is
no `tpXppYcpZ` suffix. `DENSE_LR`/`SPARSE_LR` are the only tunable
hyperparameters and scale linearly with global batch size.

### Seeds

`$SEED` seeds each embedding table's initializer deterministically and must
be identical across every rank for initialization to match.
