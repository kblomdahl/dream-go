DREAM_GO=/mnt/dream_go

all: libdg_go.so
	python setup.py build_ext --inplace

libdg_go.so: $(DREAM_GO)/target/release/libdg_go.so
	cp -fu "$<" "$@"

clean:
	find train_tf/ -name '*.so' -delete
	find train_tf/ -name '*.c' -delete
