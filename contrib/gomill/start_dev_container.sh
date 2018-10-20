#!/bin/sh

exec nvidia-docker run \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	-it --rm -v "`pwd`:/app" \
	-e CUDA_VISIBLE_DEVICES=0,1 \
	$(eval docker build -q .)
