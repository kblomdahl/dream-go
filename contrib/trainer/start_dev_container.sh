#!/bin/sh

(cd "../go" && cargo build --release) && \
    cp ../../target/release/libgo.so .

nvidia-docker run \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	--net host -it --rm -v "`pwd`:/app" \
	-e CUDA_VISIBLE_DEVICES=0 \
	nvcr.io/nvidia/tensorflow:18.08-py3
