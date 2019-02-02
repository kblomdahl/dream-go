#!/bin/sh

cargo build --release --frozen && \
    (cp ../../target/release/libdg_go.so . || cp ../../target/release/libdg_go.dll .)

exec nvidia-docker run \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	--net host -it --rm -v "`pwd`:/app:cached" \
	-e CUDA_VISIBLE_DEVICES=0 \
	$(eval docker build -q .)
