#!/bin/sh

exec nvidia-docker run \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	--net host -it --rm -v "`pwd`:/app:cached" \
	-u $(id -u):$(id -g) \
	-e CUDA_VISIBLE_DEVICES=0 \
	$(eval docker build -q .)
