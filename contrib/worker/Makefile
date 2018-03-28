.PHONY: build clean docker

VERSION?=0.5.0
DEVICE?=0
DB?=upload.dg.io:8080

all:

clean:
	rm -rf .staging/

build:
	mkdir -p .staging/
	cd ../.. ; cargo build --release
	cp ../../target/release/dream_go .staging/dream_go
	cp ../../tools/upload2rest.py .staging/upload2rest.py
	cp ../../tools/sgf2score.py .staging/sgf2score.py
	cp loop_worker.sh .staging/loop_worker.sh

	docker build -t "dream_go/worker:$(VERSION)" .

docker: build
	# this is for local deployments only, for a larger scale deployment you
	# will want to push the images to some repository and then deploy them from
	# there.
	# 
	# You will however still need to do the following:
	# 
	# - provide the GPU to the container as device 0 (most cloud providers will
	#   do this for you)
	# - set the environment variable `DB` to point towards a host (and port)
	#   where the `dream_go/db` image is available.
	# 
	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(DEVICE) \
		-e DB=$(DB) \
		-td --rm "dream_go/worker:$(VERSION)"
