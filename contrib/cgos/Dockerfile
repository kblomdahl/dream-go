FROM nvcr.io/nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04
RUN apt-get update --no-upgrade -yq && \
    apt-get install --no-upgrade -yq curl m4
RUN mkdir /app
WORKDIR /app
RUN curl 'http://www.yss-aya.com/cgos/software/cgosGtp-linux-x86_64.tar.gz' | tar zx
COPY .staging/dream_go /app/dream_go
COPY .staging/dream_go.json /app/dream_go.json
COPY config.txt /app/config.base.txt

ARG GIT_REV
ARG CGOS_PASSWORD
RUN m4 -D REV=$GIT_REV \
       -D PASSWORD=$CGOS_PASSWORD < /app/config.base.txt > /app/config.txt
CMD /app/cgosGtp-linux-x86_64 -c /app/config.txt
