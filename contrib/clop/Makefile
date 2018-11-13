.PHONY: all build clean

all: build bin/dream_go bin/dream_go.json

clean:
	rm -f bin/dream_go bin/dream_go.json
	rm -f games/*.sgf
	rm -f DreamGo*

build:
	cargo build --release --bin dream_go --frozen

bin/dream_go: ../../target/release/dream_go
	cp -f ../../target/release/dream_go bin/.

bin/dream_go.json: ../../dream_go.json
	cp -f ../../dream_go.json bin/.

