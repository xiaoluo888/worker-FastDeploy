ARG PADDLEOCR_VERSION=">=3.3.2,<3.4"

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install PaddlePaddle GPU 3.2.1 from official repo (CUDA 12.6 compatible with 12.4)
RUN pip install paddlepaddle-gpu==3.2.1 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/
RUN python -m pip install fastdeploy-gpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-86_89/ --extra-index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple


ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
RUN python -m pip install https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

RUN echo "Installing PaddleOCR with version constraint: ${PADDLEOCR_VERSION}" && \
    python -m pip install --upgrade pip && \
    python -m pip install "paddleocr[doc-parser]>=3.3.2,<3.4" && \
    python -m pip install "paddlex==3.3.11" && \
    python -m pip install "runpod>=1.8,<2.0" && \
    paddlex --install serving
    
# ---- create user ----
RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr

# ---- ENV: force PaddleX + HF to use offline cache ----
ENV HOME=/home/paddleocr
ENV PADDLEX_HOME=/home/paddleocr/.paddlex
ENV PADDLEX_MODEL_HOME=/home/paddleocr/.paddlex/official_models
ENV HF_HOME=/home/paddleocr/.cache/huggingface
ENV TRANSFORMERS_CACHE=/home/paddleocr/.cache/huggingface
ENV DISABLE_MODEL_SOURCE_CHECK=True

WORKDIR /home/paddleocr

# ---- offline model baking ----
ARG BUILD_FOR_OFFLINE=true
RUN if [ "${BUILD_FOR_OFFLINE}" = "true" ]; then \
    mkdir -p ${PADDLEX_MODEL_HOME} && \
    cd ${PADDLEX_MODEL_HOME} && \
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
         https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
         https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV2_infer.tar \
         https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL_infer.tar && \
    tar -xf UVDoc_infer.tar && mv UVDoc_infer UVDoc && \
    tar -xf PP-LCNet_x1_0_doc_ori_infer.tar && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori && \
    tar -xf PP-DocLayoutV2_infer.tar && mv PP-DocLayoutV2_infer PP-DocLayoutV2 && \
    tar -xf PaddleOCR-VL_infer.tar && mv PaddleOCR-VL_infer PaddleOCR-VL && \
    rm *.tar && \
    mkdir -p ${PADDLEX_HOME}/fonts && \
    wget -P ${PADDLEX_HOME}/fonts https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf; \
fi

# ---- copy code and config ----
COPY --chown=paddleocr:paddleocr /src/handler.py /src/handler.py
COPY --chown=paddleocr:paddleocr ./pipeline_config_fastdeploy.yaml /home/paddleocr/pipeline_config_fastdeploy.yaml

USER paddleocr

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/src/handler.py"]
