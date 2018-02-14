.PHONY: clean staging

VERSION?=0.5.0

all: build

clean:
	rm -rf staging/

staging:
	mkdir -p staging/
	cd ../.. ; cargo build --release
	cp ../../target/release/dream_go staging/dream_go
	cp ../../tools/upload2rest.py staging/upload2rest.py
	cp ../../tools/sgf2score.py staging/sgf2score.py
	cp loop_worker.sh staging/loop_worker.sh

build: staging
	docker build -t "dream_go/worker:$(VERSION)" .

run:
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
	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
		-e DB=upload.dg.io:8080 --link dream_db:upload.dg.io \
		-td --rm "dream_go/worker:$(VERSION)"
	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=1 \
		-e DB=upload.dg.io:8080 --link dream_db:upload.dg.io \
		-td --rm "dream_go/worker:$(VERSION)"
