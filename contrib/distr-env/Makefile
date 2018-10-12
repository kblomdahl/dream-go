.PHONY: all staging clean

all: staging
	GIT_REV=`git show-ref -s HEAD` docker-compose build

clean:
	rm -rf .staging

# -------- Staging --------

TRAIN=requirements.txt dream_tf libgo.so
WORKER=code

staging: .staging $(WORKER:%=.staging/worker/%) $(TRAIN:%=.staging/train/%)

.staging:
	mkdir -p .staging/

.staging/train/libgo.so: ../../target/release/libgo.so
	cd ../go && cargo build --quiet --release
	mkdir -p `dirname $@`
	cp -uf "$^" "$@"

.staging/train/%: ../trainer/%
	mkdir -p `dirname $@`
	cp -urf "$^" "$@"

.staging/worker/code: ../../
	rm -rf "$@"
	git clone "$^" "$@"
