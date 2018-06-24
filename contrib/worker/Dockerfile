FROM nvidia/cuda:9.1-cudnn7-devel
RUN apt-get update -y && apt-get install -y curl python3 gnugo jq gcc

RUN mkdir -p /app/
COPY .staging/upload2rest.py /app/upload2rest.py
COPY .staging/sgf2score.py /app/sgf2score.py
COPY .staging/loop_worker.sh /app/loop_worker.sh

COPY .staging/code /app/code
RUN export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs/ && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly && \
    cd /app/code/ && \
    ~/.cargo/bin/cargo build --release && \
    cp target/release/dream_go /app/dream_go

ENV OPTS "--num-samples 2 --batch-size 256 --num-games 256 --num-threads 256 --num-rollout 6400"
ENV N "1000"

WORKDIR /app
ENTRYPOINT ["/app/loop_worker.sh"]
