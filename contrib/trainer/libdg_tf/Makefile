IMAGE=$(shell docker build -q .)
TF_RUN=docker run --rm -e TF_CPP_MIN_LOG_LEVEL=3 $(IMAGE)
TF_CFLAGS=$(shell $(TF_RUN) python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LDFLAGS=$(shell $(TF_RUN) python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

all: libdg_tf.so

libdg_tf.so: sgf_to_features.cc num_channels.cc tensor_to_image.cc libdg_pipe.so
	docker run \
		--rm \
		-u 1000:1000 \
		-v "$(shell pwd):/workspace" \
		-w "/workspace" \
		$(IMAGE) \
		g++ -std=c++14 -shared $(filter-out libdg_pipe.so, $+) -o "$@" -fPIC -O2 -g \
		$(TF_CFLAGS) $(TF_LDFLAGS) \
		-L. -ldg_pipe

libdg_pipe.so: ../libdg_pipe.so
	cp -fu "$<" "$@"

clean:
	rm -f libdg_pipe.so libdg_tf.so
