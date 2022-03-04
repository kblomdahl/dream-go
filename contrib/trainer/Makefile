CARGO=cargo

all: libdg_pipe.so libdg_tf/libdg_tf.so

test:
	python -m pyflakes dream_tf/
	CUDA_VISIBLE_DEVICES= python3 -m unittest

test-gpu:
	python3 -m unittest

clean:
	rm -f libdg_pipe.so
	make -C libdg_tf clean

../../target/release/libdg_pipe.so:
	$(CARGO) build --release --frozen --lib

libdg_pipe.so: ../../target/release/libdg_pipe.so
	cp -fu "$<" "$@"

libdg_tf/libdg_tf.so: libdg_pipe.so libdg_tf/sgf_to_features.cc libdg_tf/num_channels.cc libdg_tf/tensor_to_image.cc
	$(MAKE) -C libdg_tf all
