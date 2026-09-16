export CUDA_GRAPH_IMPLEMENTATION=local
export CUDA_GRAPH_SCOPE=full_iteration
export PYTORCH_ALLOC_CONF="expandable_segments:True,graph_capture_record_stream_reuse:True"
# Setting default value for padding here. Configs can override.
export MOE_EXPERT_RANK_CAPACITY_FACTOR=2
# We get illegal memory accesses if we don't exclude NCCL from the graph
export NCCL_GRAPH_REGISTER=0
