FROM tensorflow/tensorflow:latest-gpu-py3
EXPOSE 6006

RUN mkdir -p /app /app/models
COPY .staging/upload2rest.py /app/upload2rest.py
COPY dream_tf /app/dream_tf
COPY requirements.txt /app/requirements.txt
COPY Makefile /app/Makefile
COPY loop_trainer.sh /app/loop_trainer.sh

RUN cd /app && make all

WORKDIR /app
ENTRYPOINT ["/app/loop_trainer.sh"]
