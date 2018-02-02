FROM nvidia/cuda:9.1-cudnn7-runtime
RUN apt-get update -y && apt-get install -y python3 curl

RUN mkdir -p /app/
COPY staging/dream_go /app/dream_go
COPY staging/upload2rest.py /app/upload2rest.py
COPY staging/loop_worker.sh /app/loop_worker.sh

ENV OPTS "--num-samples 4 --batch-size 16 --num-games 1 --num-threads 16"
ENV N "1"

WORKDIR /app
ENTRYPOINT ["/app/loop_worker.sh"]
