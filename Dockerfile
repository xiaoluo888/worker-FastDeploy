# ========================= 使用百度基础镜像构建新的FD环境 =========================
# FROM nvidia/cuda:12.6.0-base-ubuntu22.04 

# RUN apt-get update -y \
#     && apt-get install -y python3-pip

# RUN ldconfig /usr/local/cuda-12.6/compat/

# # Install Python dependencies
# COPY builder/requirements.txt /requirements.txt
# RUN --mount=type=cache,target=/root/.cache/pip \
#     python3 -m pip install --upgrade pip && \
#     python3 -m pip install paddlepaddle-gpu==3.2.2 -i https://www.paddlepaddle.org.cn/packages/stable/cu126/ && \
#     python3 -m pip install fastdeploy-gpu==2.3.0 -i https://www.paddlepaddle.org.cn/packages/stable/fastdeploy-gpu-80_90/ 

# # 避免交互 & 打印不缓冲
# ENV DEBIAN_FRONTEND=noninteractive \
#     PYTHONUNBUFFERED=1 \
#     PIP_DISABLE_PIP_VERSION_CHECK=1

# # 可选：加一些系统工具（调试日志/证书等）
# RUN apt-get update && apt-get install -y --no-install-recommends \
#     ca-certificates curl tini && \
#     rm -rf /var/lib/apt/lists/*

# # install runpod（Serverless SDK）
# RUN pip install --no-cache-dir runpod


# ========================= 使用自己预先构建的镜像 =========================
FROM xiaoluo888/worker-fastdeploy:latest
ARG PADDLEOCR_VERSION=">=3.3.2,<3.4"
RUN python -m pip install "paddleocr[doc-parser]${PADDLEOCR_VERSION}" \
    && paddlex --install serving

RUN groupadd -g 1000 paddleocr \
    && useradd -m -s /bin/bash -u 1000 -g 1000 paddleocr
ENV HOME=/home/paddleocr
WORKDIR /home/paddleocr


ARG BUILD_FOR_OFFLINE=true
RUN if [ "${BUILD_FOR_OFFLINE}" = 'true' ]; then \
        mkdir -p "${HOME}/.paddlex/official_models" \
        && cd "${HOME}/.paddlex/official_models" \
        && wget https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/UVDoc_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-LCNet_x1_0_doc_ori_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PP-DocLayoutV2_infer.tar \
            https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0/PaddleOCR-VL_infer.tar \
        && tar -xf UVDoc_infer.tar \
        && mv UVDoc_infer UVDoc \
        && tar -xf PP-LCNet_x1_0_doc_ori_infer.tar \
        && mv PP-LCNet_x1_0_doc_ori_infer PP-LCNet_x1_0_doc_ori \
        && tar -xf PP-DocLayoutV2_infer.tar \
        && mv PP-DocLayoutV2_infer PP-DocLayoutV2 \
        && tar -xf PaddleOCR-VL_infer.tar \
        && mv PaddleOCR-VL_infer PaddleOCR-VL \
        && rm -f UVDoc_infer.tar PP-LCNet_x1_0_doc_ori_infer.tar PP-DocLayoutV2_infer.tar PaddleOCR-VL_infer.tar \
        && mkdir -p "${HOME}/.paddlex/fonts" \
        && wget -P "${HOME}/.paddlex/fonts" https://paddle-model-ecology.bj.bcebos.com/paddlex/PaddleX3.0/fonts/PingFang-SC-Regular.ttf; \
    fi

COPY --chown=paddleocr:paddleocr pipeline_config_fastdeploy.yaml /home/paddleocr

USER paddleocr
EXPOSE 8180

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/src/handler.py"]
CMD ["paddlex", "--serve", "--pipeline", "/home/paddleocr/pipeline_config_vllm.yaml"]
