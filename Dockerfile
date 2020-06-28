FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04
EXPOSE 6006

ENV CARGO_HOME /usr/local/rustup
ENV RUSTUP_HOME /usr/local/rustup
ENV LIBRARY_PATH /usr/local/cuda/lib64/
ENV DEBIAN_FRONTEND noninteractive
ENV PATH /usr/local/rustup/bin:$PATH
ENV LD_LIBRARY_PATH /workspaces/dream-go/target/release:$LD_LIBRARY_PATH

COPY requirements.txt /tmp/requirements.txt
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y --no-install-recommends \
        curl \
        git \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        python3-venv \
        python3-wheel \
        ssh-client \
        sudo \
        zsh \
        && \
    pip3 install --upgrade pip && \
    python3 -m pip install -r /tmp/requirements.txt && \
    export RUST_VERSION=nightly-$(curl -s https://rust-lang.github.io/rustup-components-history/x86_64-unknown-linux-gnu/rls) && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain ${RUST_VERSION} && \
    rustup component add rls rust-analysis rust-src rustfmt --toolchain ${RUST_VERSION}

ENV CARGO_HOME=
