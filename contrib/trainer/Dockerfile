FROM nvcr.io/nvidia/tensorflow:20.02-tf1-py3
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/usr/local/lib/python3.6/dist-packages/tensorflow_core"
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/app/libdg_tf"
ENV LD_LIBRARY_PATH "${LD_LIBRARY_PATH}:/app"

EXPOSE 6006
WORKDIR /app
