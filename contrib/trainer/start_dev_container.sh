#!/bin/sh

nvidia-docker run \
	-it --rm \
	--net host \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	-u $(id -u):$(id -g) \
	-v "$HOME:/home/user:cached" -e "HOME=/home/user" \
	-v "$(pwd)/../..:/mnt/dream_go:cached" \
	-v "$(pwd):/app:cached" \
	-e "TF_CPP_MIN_LOG_LEVEL=1" \
	-e "TF_ENABLE_AUTO_MIXED_PRECISION=1" \
	-e "TF_FORCE_GPU_ALLOW_GROWTH=true" \
	$(eval docker build -q .) \
	$SHELL
