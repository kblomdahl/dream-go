FROM nvcr.io/nvidia/tensorflow:20.02-tf1-py3
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

EXPOSE 6006
WORKDIR /app
