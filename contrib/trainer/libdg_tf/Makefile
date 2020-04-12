IMAGE=$(shell docker build -q .)
TF_RUN=docker run --rm -e TF_CPP_MIN_LOG_LEVEL=3 $(IMAGE)
TF_CFLAGS=$(shell $(TF_RUN) python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS=$(shell $(TF_RUN) python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

all: libdg_tf.so

libdg_tf.so: sgf_to_features.cc tensor_to_image.cc libdg_go.so
	docker run \
		--rm \
		-u 1000:1000 \
		-v "$(shell pwd):/workspace" \
		$(IMAGE) \
		g++ -std=c++11 -shared $(filter-out libdg_go.so, $+) -o "$@" -fPIC -O2 -g \
		$(TF_CFLAGS) $(TF_LDFLAGS) \
		-L. -ldg_go

libdg_go.so: ../libdg_go.so
	cp -fu "$<" "$@"

clean:
	rm -f libdg_go.so libdg_tf.so 

