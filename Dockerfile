FROM ccr-2vdh3abv-pub.cnc.bj.baidubce.com/paddlepaddle/paddleocr-genai-fastdeploy-server:latest-offline


# Copy handler
COPY /src/handler.py /src/handler.py
CMD ["python3", "/src/handler.py"]
