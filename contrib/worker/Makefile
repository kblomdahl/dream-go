VERSION?=0.5.0

.PHONY: all clean

all:
	mkdir -p .staging/
	rm -rf .staging/code/ && git clone ../.. .staging/code/
	cp ../../tools/upload2rest.py .staging/upload2rest.py
	cp ../../tools/sgf2score.py .staging/sgf2score.py
	cp loop_worker.sh .staging/loop_worker.sh

	docker build -t "dream_go/worker:$(VERSION)" .

clean:
	rm -rf .staging/
