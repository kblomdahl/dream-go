FROM tensorflow/tensorflow:latest-gpu-py3
EXPOSE 6006

RUN curl -sL https://github.com/stedolan/jq/releases/download/jq-1.5/jq-linux64 -o /usr/bin/jq
RUN chmod +x /usr/bin/jq

RUN mkdir -p /app /app/models
COPY .staging/upload2rest.py /app/upload2rest.py
COPY dream_tf /app/dream_tf
COPY requirements.txt /app/requirements.txt
COPY Makefile /app/Makefile
COPY loop_trainer.sh /app/loop_trainer.sh

RUN cd /app && pip3 install -r requirements.txt

WORKDIR /app
ENTRYPOINT ["/app/loop_trainer.sh"]
