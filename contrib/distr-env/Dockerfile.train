FROM dream_go/base:latest
FROM nvcr.io/nvidia/tensorflow:18.12-py3
COPY requirements.txt /tmp/requirements.txt
RUN pip install -q -r /tmp/requirements.txt
RUN mkdir -p /app /app/data /app/models

COPY .staging/train/. /app/
COPY --from=0 /app/libdg_go.so /app/libdg_go.so
COPY dg_storage.py /app/dg_storage.py
COPY run_train.py /app/run_train.py
COPY google-storage-auth.json /app/google-storage-auth.json
RUN pip install -q -r /app/requirements.txt

ARG GIT_REV
ENV GOOGLE_APPLICATION_CREDENTIALS /app/google-storage-auth.json
ENV GIT_REV $GIT_REV

WORKDIR /app
CMD /app/run_train.py
