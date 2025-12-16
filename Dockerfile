FROM xiaoluo888/worker-fastdeploy:latest

ARG PADDLEOCR_VERSION=">=3.3.2,<3.4"
RUN echo "Installing PaddleOCR with version constraint: ${PADDLEOCR_VERSION}" && \
    python -m pip install --upgrade pip && \
    python -m pip install "paddleocr[doc-parser]>=3.3.2,<3.4" && \
    
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
ENV PYTHONUNBUFFERED=1

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

# ---- copy code ----
COPY --chown=paddleocr:paddleocr /src/handler.py /src/handler.py

USER paddleocr

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/src/handler.py"]
