VERSION?=0.5.0

.PHONY: all clean

all:
	mkdir -p .staging/
	rm -rf .staging/code/ && git clone ../.. .staging/code/
	cp ../../tools/sgf2big.py .staging/sgf2big.py
	cp ../../tools/sgf2elo.py .staging/sgf2elo.py
	cp ../../tools/sgf2score.py .staging/sgf2score.py
	cp ../../tools/upload2rest.py .staging/upload2rest.py
	cp gomill-playoff.ctl.py .staging/gomill-playoff.ctl.py
	cp loop_worker.sh .staging/loop_worker.sh

	docker build -t "dream_go/worker:$(VERSION)" .

clean:
	rm -rf .staging/
