#!/bin/sh

(cd "../go" && cargo build --release) && \
    (cp ../../target/release/libgo.so . || cp ../../target/release/go.dll .)

exec nvidia-docker run \
	--shm-size=1g --ulimit memlock=-1 --ulimit stack=67108864 \
	--net host -it --rm -v "`pwd`:/app" \
	-e CUDA_VISIBLE_DEVICES=0 \
	$(eval docker build -q .)
