FROM nvidia/cuda:9.0-cudnn7-runtime
RUN apt-get update -y && apt-get install -y curl python3 python3-pip && pip3 install tensorflow-gpu scipy
EXPOSE 6006

CMD mkdir -p /app/ /app/logs /app/models
COPY staging/bootstrap.py /app/bootstrap.py
COPY staging/upload2rest.py /app/upload2rest.py
COPY staging/loop_trainer.sh /app/loop_trainer.sh

WORKDIR /app
ENTRYPOINT ["/app/loop_trainer.sh"]
