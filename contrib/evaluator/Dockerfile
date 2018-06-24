FROM nvidia/cuda:9.1-cudnn7-devel
RUN apt-get update -y && apt-get install -y \
    curl python python-pip python3 python3-pip jq gcc

RUN mkdir -p /app/
COPY .staging/sgf2big.py /app/sgf2big.py
COPY .staging/sgf2elo.py /app/sgf2elo.py
COPY .staging/upload2rest.py /app/upload2rest.py
COPY .staging/requirements.txt /app/requirements.txt
COPY .staging/gomill-playoff.ctl.py /app/gomill-playoff.ctl.py
COPY .staging/loop_eval.sh /app/loop_eval.sh

COPY .staging/code /app/code
RUN export LIBRARY_PATH=/usr/local/cuda/lib64/:/usr/local/cuda/lib64/stubs/ && \
    curl https://sh.rustup.rs -sSf | sh -s -- -y --default-toolchain nightly && \
    cd /app/code/ && \
    ~/.cargo/bin/cargo build --release && \
    cp target/release/dream_go /app/dream_go
RUN pip3 install -r /app/requirements.txt
RUN pip install gomill

ENV TZ "UTC"
ENV N "5"

WORKDIR /app
ENTRYPOINT ["/app/loop_eval.sh"]
