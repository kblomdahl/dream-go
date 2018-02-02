.PHONY: create-volume staging

VERSION?=0.5.0

clean:
	make -C database clean
	make -C worker clean

build: clean
	make -C database build clean
	make -C worker build clean

run:
	make -C database run
	make -C worker run
