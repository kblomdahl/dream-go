CARGO=cargo

all: libdg_go.so libdg_tf/libdg_tf.so

test:
	python3 -m unittest

clean:
	rm -f libdg_go.so
	make -C libdg_tf clean

../../target/release/libdg_go.so:
	$(CARGO) build --release --frozen --lib

libdg_go.so: ../../target/release/libdg_go.so
	cp -fu "$<" "$@"

libdg_tf/libdg_tf.so: libdg_go.so libdg_tf/sgf_to_features.cc libdg_tf/tensor_to_image.cc
	$(MAKE) -C libdg_tf all
