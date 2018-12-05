.PHONY: build clean init

all: build

build:
	python setup.py build_ext --inplace

clean:
	find dream_net/ -type f -name '*.c' -delete
	find dream_net/ -type f -name '*.so' -delete
	find dream_net/ -type d -name '__pycache__' -prune -exec rm -rf '{}' ';'
	rm -rf build/

init:
	pip install -r requirements.txt
