#FROM tensorflow/tensorflow:1.13.1-gpu-py3
FROM nvcr.io/nvidia/tensorflow:19.05-py3
RUN echo "/app" > /etc/ld.so.conf.d/app.conf && \
    ldconfig

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    apt-get update && \
    apt-get install -y zsh

EXPOSE 6006
WORKDIR /app
