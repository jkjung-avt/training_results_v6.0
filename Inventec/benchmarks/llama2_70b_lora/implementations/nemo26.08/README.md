## Running LLama2-70B LoRA MLPerf Training Benchmark

This file contains the instructions for running the NVIDIA NeMo LLama2-70B LoRA MLPerf Benchmark on Inventec GPU servers.

## 1. Environment Setup

- Refer to [ENV_SETUP.md](../../../../ENV_SETUP.md).
- Assuming the source code has been cloned at `/mnt/jkjung/training_results_v6.0`.

## 2. Set up

### 2.1 Build the container and the SquashFS file

```bash
docker build -t mlperf-inventec:llama2_70b_lora_nemo26.08 .
enroot import -o /mnt/sqsh/llama2_70b_lora_nemo26.08.sqsh dockerd://mlperf-inventec:llama2_70b_lora_nemo26.08
```

### 2.2 Download dataset and model + preprocessing

This benchmark uses the [GovReport](https://gov-report-data.github.io/) dataset.

The dataset download/preprocessing scripts are included in the container. To invoke them, you need either a docker or slurm/enroot environment. Start the container.

```bash
docker run -it --rm --gpus=all --network=host --ipc=host --volume /raid/data/mlperf_training/llama2_70b_lora:/data mlperf-inventec:llama2_70b_lora_nemo26.08

# now you should be inside the container in the /workspace/ft-llm directory
python scripts/download_dataset.py --data_dir /data/gov_report  # download and preprocess dataset; takes less than 1 minute
python scripts/download_model.py --model_dir /data/model        # download and preprocess model checkpoint used for initialization; could take up to 30 minutes
```

After both scripts finish you should see the following files in the `/data` directory:

```
/data
├── gov_report
│   ├── train_metadata.jsonl
│   ├── train.npy
│   ├── validation_metadata.jsonl
│   └── validation.npy
└── model
    ├── iter_0000000
    │   ├── __0_0.distcp
    │   ├── __0_1.distcp
    │   ├── common.pt
    │   ├── metadata.json
    │   ├── modelopt_run_config.yaml
    │   ├── run_config.yaml
    │   ├── tokenizer
    │   │   ├── special_tokens_map.json
    │   │   ├── tokenizer_config.json
    │   │   ├── tokenizer.json
    │   │   └── tokenizer.model
    │   └── train_state.pt
    ├── latest_checkpointed_iteration.txt
    └── latest_train_state.pt

5 directories, 17 files
```

Exit the container.

## 3. Launch training

For training, we use Slurm with the Pyxis extension, and Slurm's MPI support to run our container.

Navigate to the directory where `run.sub` is stored.

The launch command structure:

```bash
export CONT=/mnt/sqsh/llama2_70b_lora_nemo26.08.sqsh
export LOGDIR=../../../../results/P5800G7_nemo26.08/llama2_70b_lora
export MODEL=/raid/data/mlperf_training/llama2_70b_lora/model
export DATADIR=/raid/data/mlperf_training/llama2_70b_lora/gov_report
source config_P5800G7_1x8x1xtp1pp1cp1_fp4.sh
```

Launch the training job on a specific compute node (e.g. p5800-1):

```bash
sbatch -w p5800-1 --time=${WALLTIME} run.sub
```

Or just launch the training job on any idle compute node:


```bash
sbatch -N ${DGXNNODES} --time=${WALLTIME} run.sub
```

All configuration files follow the format `config_<SYSTEM_NAME>_<NODES>x<GPUS/NODE>x<BATCH/GPU>xtpXppYcpZ.sh`, where X represents tensor parallel, Y represents pipeline parallel, and Z represents context parallel.

## 4. Evaluation

### Quality metric
Cross entropy loss

### Quality target
0.925

### Evaluation frequency
Every 384 sequences, CEIL(384 / global_batch_size) steps if 384 is not divisible by GBS. Skipping first FLOOR(0.125*global_batch_size+2) evaluations

### Evaluation thoroughness
Evaluation on the validation subset that consists of 173 examples
