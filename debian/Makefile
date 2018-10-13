#!/usr/bin/make -f

VERSION?=0.6.1
DIR=dreamgo-$(VERSION)_amd64

.PHONY: clean build

all: $(DIR).deb $(DIR).tar.gz

zip: $(DIR).zip

clean:
	rm -rf $(DIR)

build:
	cargo build --release

install: build control
	rm -rf $(DIR)
	mkdir -p $(DIR)/DEBIAN/
	mkdir -p $(DIR)/usr/games/
	mkdir -p $(DIR)/usr/share/dreamgo/

	cp -f ../target/release/dream_go $(DIR)/usr/games/
	cp -f ../dream_go.json $(DIR)/usr/share/dreamgo/
	cp -f control $(DIR)/DEBIAN/

$(DIR).deb: install
	dpkg-deb --build $(DIR)

$(DIR).tar.gz: build
	rm -rf $(DIR) && mkdir -p $(DIR)
	cp -f ../target/release/dream_go $(DIR)/
	cp -f ../dream_go.json $(DIR)/

	tar czf $(DIR).tar.gz $(DIR)/

$(DIR).zip: build
	rm -rf $(DIR) && mkdir -p $(DIR)
	cp -f ../target/release/dream_go.exe $(DIR)/
	cp -f ../dream_go.json $(DIR)/

	curl http://docs.nvidia.com/deeplearning/sdk/pdf/cuDNN-SLA.pdf -o $(DIR)/cuDNN-SLA.pdf
	curl http://docs.nvidia.com/cuda/pdf/EULA.pdf -o $(DIR)/CUDA-EULA.pdf

	zip $(DIR).zip $(DIR)/*
