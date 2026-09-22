# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/pytorch:26.08-py3
FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=${GIT_COMMIT_ID}

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV PIP_CONSTRAINT=""

WORKDIR /workspace/dlrmv4

## fbgemm_gpu — main tip as of 2026-07-11 (last known-good fallback: the
## mlcommons/training#889-validated 10b775730212923f65f7b78f79b6a01d80cf3c29).
## Release tags up to v1.5.0 lack fbgemm_gpu.tbe.monitoring, which the
## torchrec tag below imports. The nvml stub: docker build has no driver.
ARG FBGEMM_REVISION=a5941c166797e2ee4d5bb2ece72a33e4b8adaac1
# Rubin (VR200) is sm_107 and does not load plain sm_100 SASS, so fbgemm needs
# real sm_107 kernels (nvcc >= 13.4; the base torch itself runs via PTX JIT).
#ARG FBGEMM_CUDA_ARCH_LIST="10.0;10.7"
ARG FBGEMM_CUDA_ARCH_LIST="10.0;10.3"
RUN git init FBGEMM && \
    cd FBGEMM && \
    git remote add origin https://github.com/pytorch/FBGEMM.git && \
    git fetch --depth 1 origin ${FBGEMM_REVISION} && \
    git checkout FETCH_HEAD && \
    git submodule update --init --depth 1 \
        external/asmjit external/cpuinfo external/cutlass external/json && \
    pip install -r fbgemm_gpu/requirements.txt && \
    # Patch for the CUDA 13.4 cudafe++ bug:
    sed -i 's/^using namespace at;$/using at::Device;\nusing at::IntArrayRef;\nusing at::kCPU;\nusing at::Storage;\nusing at::Tensor;/' \
        fbgemm_gpu/src/memory_utils/memory_utils.cu && \
    grep -q "using at::Tensor;" fbgemm_gpu/src/memory_utils/memory_utils.cu && \
    cd fbgemm_gpu && \
    TORCH_CUDA_ARCH_LIST="${FBGEMM_CUDA_ARCH_LIST}" \
    python setup.py install --build-target=default \
        --nvml_lib_path /usr/local/cuda/lib64/stubs/libnvidia-ml.so

## TorchRec
ARG TORCHREC_REVISION=v2026.06.01.00
ENV CUSTOM_TORCHREC_REVISION=${TORCHREC_REVISION}
RUN if [ "${TORCHREC_REVISION}" != SKIP ]; then \
    pip install --no-deps "git+https://github.com/pytorch/torchrec.git@${TORCHREC_REVISION}" \
    ; fi

COPY requirements.txt .
RUN apt-get remove -y python3-yaml && pip install -r requirements.txt

## cuDNN Frontend HSTU attention for Blackwell/Rubin. Install without
## dependencies: this branch declares nvidia-cutlass-dsl==4.5.0, and the base
## image already provides a compatible CuTe DSL build.
ARG CUDNN_FRONTEND_REVISION=develop
RUN pip install --no-deps --force-reinstall "git+https://github.com/NVIDIA/cudnn-frontend.git@${CUDNN_FRONTEND_REVISION}"

# tensordict 0.8.3 (upstream pins 0.6.2, which has no linux-aarch64 wheel)
RUN pip install --no-deps \
        torchmetrics==1.0.3 \
        tensordict==0.8.3 \
        "git+https://github.com/mlcommons/logging.git@d197741894957ff281b7b411f256d14581b74f0b"

COPY . .

# import smoke test; fbgemm_gpu needs libnvidia-ml.so.1, shim it with the stub
RUN mkdir -p /tmp/nvml-stub && \
    ln -s /usr/local/cuda/lib64/stubs/libnvidia-ml.so /tmp/nvml-stub/libnvidia-ml.so.1 && \
    export LD_LIBRARY_PATH="/tmp/nvml-stub:${LD_LIBRARY_PATH}" && \
    python -c "import torch, fbgemm_gpu, torchrec, torchmetrics, tensordict, polars, xxhash, gin, tvm_ffi" && \
    python -c "from mlperf_logging import mllog" && \
    python -c "import generative_recommenders.dlrm_v4.train.mlperf_logging_utils" && \
    rm -rf /tmp/nvml-stub
RUN python -c "from cudnn import hstu_attention_backward, hstu_attention_forward"

