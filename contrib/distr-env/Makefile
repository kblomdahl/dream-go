.PHONY: all staging clean

all: staging
	GIT_REV=`git show-ref --head --tags -s HEAD` docker-compose build

clean:
	docker-compose down --rmi 'all'
	rm -rf .staging

# -------- Staging --------

TRAIN=requirements.txt dream_tf

staging: .staging .staging/code $(WORKER:%=.staging/worker/%) $(TRAIN:%=.staging/train/%)

.staging:
	mkdir -p .staging/

.staging/train/%: ../trainer/%
	mkdir -p `dirname $@`
	cp -urf "$^" "$@"

.staging/code: ../../
	rm -rf "$@"
	git clone "$^" "$@"
