FROM nvidia/cuda:9.1-cudnn7-runtime
RUN apt-get update -y && apt-get install -y curl python3

RUN mkdir -p /app/
COPY staging/dream_go /app/dream_go
COPY staging/upload2rest.py /app/upload2rest.py
COPY staging/loop_worker.sh /app/loop_worker.sh

ENV OPTS "--num-samples 1 --batch-size 256 --num-games 256 --num-threads 256"
ENV N "1000"

WORKDIR /app
ENTRYPOINT ["/app/loop_worker.sh"]
