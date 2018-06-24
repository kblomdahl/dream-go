DEVICE?=1
DB?=localhost:3000
VERSION?=0.5.0

.PHONY: all clean

all:
	mkdir -p .staging/
	rm -rf .staging/code/ && git clone ../.. .staging/code/
	cp ../../tools/sgf2big.py .staging/sgf2big.py
	cp ../../tools/sgf2elo.py .staging/sgf2elo.py
	cp ../../tools/upload2rest.py .staging/upload2rest.py
	cp requirements.txt .staging/requirements.txt
	cp gomill-playoff.ctl.py .staging/gomill-playoff.ctl.py
	cp loop_eval.sh .staging/loop_eval.sh

	docker build -t "dream_go/evaluator:$(VERSION)" .

clean:
	rm -rf .staging/

run:
	docker run --runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=$(DEVICE) \
		-e DB=$(DB) --net=host \
		-td --rm "dream_go/evaluator:$(VERSION)"
