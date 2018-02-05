.PHONY: create-volume staging

VERSION?=0.5.0

clean:
	make -C database clean
	make -C worker clean
	make -C trainer clean

build: clean
	make -C database build clean
	make -C worker build clean
	make -C trainer build clean

run:
	make -C database run
	make -C worker run
	make -C trainer run
