# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

ARG FROM_IMAGE_NAME=nvcr.io/nvidia/nemo:26.08
FROM ${FROM_IMAGE_NAME}

ARG GIT_COMMIT_ID
ENV GIT_COMMIT_ID=$GIT_COMMIT_ID

RUN git config --global user.name "a" && \
    git config --global user.email "a"

ENV NVIDIA_PRODUCT_NAME=PyTorch
ENV CUDNN_VERSION=9.21.1.3
ENV NCCL_VERSION=2.30.7-1
ENV PIP_CONSTRAINT=""
#ENV PYTHONPATH="/opt/Megatron-Bridge/src:/opt/Megatron-Bridge/3rdparty/Megatron-LM"

RUN pip install numcodecs==0.16.5 && \
    pip install nvidia-mathdx==25.6.0

WORKDIR /workspace/

#ENV TE_REVISION=2.16.0+4220403e
ARG TE_REVISION=v2.16.1
ENV CUSTOM_TE_REVISION=${TE_REVISION}
RUN if [ "${TE_REVISION}" != SKIP ]; then \
    pip uninstall -y transformer-engine && \
    git clone https://github.com/NVIDIA/TransformerEngine.git transformerengine && \
    cd transformerengine && \
    git checkout ${TE_REVISION} && \
    echo TE_COMMIT_HASH=$(git rev-parse HEAD) && \
    echo $(git rev-parse HEAD) > /TE_COMMIT_HASH.env && \
    git submodule init && git submodule update && \
    NVTE_CUDA_ARCHS="100a;103a" NVTE_UB_WITH_MPI=1 NVTE_FRAMEWORK=pytorch MPI_HOME=/usr/local/mpi pip install --no-build-isolation . \
    ; fi

ARG CPATH=/usr/local/cuda-13.3/targets/x86_64-linux/include/cccl
ARG LIBRARY_PATH=/usr/local/cuda/lib64/stubs:$LIBRARY_PATH
RUN ln -s /usr/local/cuda/lib64/stubs/libnvidia-ml.so /usr/local/cuda/lib64/stubs/libnvidia-ml.so.1 && \
    git clone --branch hybrid-ep https://github.com/deepseek-ai/DeepEP.git && \
    cd DeepEP && git checkout 10d4dd7377d5bce900fbb4b80cce863764892b95 && \
    TORCH_CUDA_ARCH_LIST="10.0;10.3" pip install --no-build-isolation .

#COPY requirements.txt .
#RUN pip uninstall -y transformers torchao && pip install --no-cache-dir -r requirements.txt

RUN pip install git+https://github.com/mlcommons/logging.git@6.0.0-rc6 && \
    pip install git+https://github.com/NVIDIA/mlperf-common.git@training-v6.0 && \
    pip install hydra-core==1.3.5

WORKDIR /workspace/llm
COPY . .
RUN cd /workspace/llm/embedding_lib && pip install --force-reinstall --no-deps --no-build-isolation .
