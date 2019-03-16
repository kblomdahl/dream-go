FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update --no-upgrade -yq && \
    apt-get install --no-upgrade -yq python python-pip && \
    pip install gomill

WORKDIR /app
CMD /bin/bash
