.PHONY: clean staging

VERSION?=0.5.0

all: build

clean:
	rm -rf staging/

staging:
	mkdir -p staging/
	cp ../../tools/bootstrap.py staging/bootstrap.py
	cp ../../tools/upload2rest.py staging/upload2rest.py
	cp loop_trainer.sh staging/loop_trainer.sh

build: staging
	docker build -t "dream_go/trainer:$(VERSION)" .

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
	# Optional (for monitoring the process):
	# - add some persistent (?) storage to the `app/models` directory
	# - add some persistent (?) storage to the `app/logs` directory
	# 
	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=0 \
		-v "$(realpath ../../data):/mnt/data" \
		-v "$(realpath ../../models):/app/models" \
		-v "$(realpath ../../logs):/app/logs" \
		-p 6006:6006 -e DB=upload.dg.io:8080 --link dream_db:upload.dg.io \
		-td --rm "dream_go/trainer:$(VERSION)"
