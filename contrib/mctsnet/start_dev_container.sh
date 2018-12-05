#!/bin/sh

OPTS="--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864"

docker build . && \
    nvidia-docker run -it $OPTS --net host --rm -v "$PWD:/app" $(eval docker build -q .)
