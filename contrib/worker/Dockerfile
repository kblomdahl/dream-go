FROM nvidia/cuda:9.1-cudnn7-devel
RUN apt-get update -y && apt-get install -y \
    curl gnugo jq gcc libblosc-dev \
    python3 python3-pip python python-pip

RUN mkdir -p /app/
COPY .staging/sgf2big.py /app/sgf2big.py
COPY .staging/sgf2elo.py /app/sgf2elo.py
COPY .staging/sgf2score.py /app/sgf2score.py
COPY .staging/upload2rest.py /app/upload2rest.py
COPY .staging/gomill-playoff.ctl.py /app/gomill-playoff.ctl.py
COPY .staging/loop_worker.sh /app/loop_worker.sh

COPY .staging/code /app/code
RUN export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs/ && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly && \
    cd /app/code/ && \
    ~/.cargo/bin/cargo build --release && \
    cp target/release/dream_go /app/dream_go
RUN pip3 install numpy scipy && pip install gomill

ENV TZ "UTC"
ENV OPTS "--num-samples 2 --batch-size 256 --num-games 256 --num-threads 256 --num-rollout 3200"
ENV N "1000"

WORKDIR /app
ENTRYPOINT ["/app/loop_worker.sh"]
