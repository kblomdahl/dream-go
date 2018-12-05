FROM nvcr.io/nvidia/tensorflow:18.08-py3
COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && pip install -r /tmp/requirements.txt

WORKDIR /app
