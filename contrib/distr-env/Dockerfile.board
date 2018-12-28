FROM nvcr.io/nvidia/tensorflow:18.12-py3
RUN mkdir -p /app /app/models

EXPOSE 6006
WORKDIR /app
CMD tensorboard --logdir models/
