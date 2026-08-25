# Environment Setup

Inventec AI Lab Environment Setup for MLPerf Training
-----------------------------------------------------

* Cluster

  - NVIDIA Base Command Manager (BCM 11) based GPU server cluster
  - BCM 11 head nodes * 2 (K888G7)
  - GPU servers (P5800G7, P9000G7, etc.), with OS images deployed from BCM head node
  - NFS server
  - WekaIO servers (5+2+1 configuration)

* Slurm

  - Slurm is installed as BCM's workload manager
  - Slurm client/compute nodes: p5800-1, p9000-1, etc.
  - Slurm server/head nodes: BCM head nodes also act as Slurm server nodes

* Storage on the compute nodes

  - `/`: local M.2 SSD, for OS image
  - `/raid`: RAID0 based on local U.2 (8x), used for MLPerf Training data (training data, validation data, tokenizer and checkpoints, etc.)
  - `/hps`: WekaIO high performance storage, as an alternative to `/raid` especially for multi-node scenarios
  - `/mnt`: NFS storage, used for source code and Docker container SquashFS files
