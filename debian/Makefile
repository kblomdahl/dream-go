#!/usr/bin/make -f

VERSION?=0.5.0
DIR=dreamgo-$(VERSION)_amd64

.PHONY: clean build

all: $(DIR).deb $(DIR).tar.gz

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
