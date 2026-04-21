# ========================= 使用官方 FastDeploy 镜像 =========================
ARG FASTDEPLOY_BASE_IMAGE=ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/fastdeploy-cuda-12.9:2.5.0
FROM ${FASTDEPLOY_BASE_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/src

WORKDIR /src

# Official image already provides PaddlePaddle + FastDeploy.
# The worker only needs the RunPod runtime and tini on top.
RUN apt-get update && apt-get install -y --no-install-recommends tini && \
    rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --no-cache-dir "runpod>=1.8,<2.0"

# If you later need broader GPU compatibility than the official image offers,
# rebuild from a plain CUDA base and reinstall matching Paddle/FastDeploy wheels.

# Overlay the current repository source so RunPod builds use this checkout,
# not whatever code happened to be baked into the base image.
COPY src/*.py /src/
COPY src/test_input.json /src/

EXPOSE 8180

ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["python3", "/src/handler.py"]
