FROM xiaoluo888/worker-fastdeploy:latest

# ---- environment ----
ENV PIP_NO_CACHE_DIR=0
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PATH=$PATH:/usr/local/bin

# ---- install system dependencies ----
USER root
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        wget \
        python3-dev \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# ---- install safetensors for CUDA 12.6 ----
RUN python3 -m pip install --no-cache-dir https://paddle-whl.bj.bcebos.com/nightly/cu126/safetensors/safetensors-0.6.2.dev0-cp38-abi3-linux_x86_64.whl

# ---- upgrade pip ----
RUN python3 -m pip install --upgrade pip

# ---- install PaddleOCR and PaddleX ----
RUN python3 -m pip install --no-cache-dir "paddleocr[doc-parser]" "paddlex==3.3.11"


# ---- create user ----
RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr

# ---- environment variables for offline cache ----
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
    mkdir -p ${PADDLEX_MODEL_HOME} && cd ${PADDLEX_MODEL_HOME} && \
    wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
         https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
         https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV2_infer.tar \
         https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL_infer.tar && \
    tar -xf UVDoc_infer.tar && mv UVDoc_infer UVDoc && \
    tar -xf PP-LCNet_x1_0_doc_ori_infer.tar && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori && \
    tar -xf PP-DocLayoutV2_infer.tar && mv PP-DocLayoutV2_infer PP-DocLayoutV2 && \
    tar -xf PaddleOCR-VL_infer.tar && mv PaddleOCR-VL_infer PaddleOCR-VL && \
    rm -f *.tar && \
    mkdir -p ${PADDLEX_HOME}/fonts && \
    wget -P ${PADDLEX_HOME}/fonts https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf; \
fi

# 🔥 THIS IS THE CRITICAL FIX
RUN mkdir -p /home/paddleocr/.paddlex && \
    chown -R paddleocr:paddleocr /home/paddleocr

USER paddleocr


ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/src/handler.py"]
