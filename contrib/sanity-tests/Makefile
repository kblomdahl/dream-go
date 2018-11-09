TESTS=$(sort $(wildcard tests/*.py))

.PHONY: all $(TESTS)

all: build dream_go dream_go.json $(TESTS)

build:
	@cargo build --quiet --release --bin dream_go --frozen

dream_go: ../../target/release/dream_go
	@cp -f $< $@

dream_go.json: ../../dream_go.json
	@cp -f $< $@

$(TESTS):
	@PYTHONPATH=. ./$@

