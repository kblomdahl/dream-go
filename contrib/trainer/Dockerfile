FROM tensorflow/tensorflow:2.0.0a0-gpu-py3
RUN echo "/app" > /etc/ld.so.conf.d/app.conf && \
    ldconfig

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt && \
    apt-get install -y zsh

EXPOSE 6006
WORKDIR /app
