FROM ubuntu
EXPOSE 8080

RUN apt-get update -y && apt-get install -y python3 python3-pip make
RUN mkdir -p /app/
COPY dream_db /app/dream_db
COPY requirements.txt /app/requirements.txt
COPY Makefile /app/Makefile

RUN cd /app && make all

WORKDIR /app
ENTRYPOINT ["/usr/bin/python3", "-m", "dream_db", "/mnt/db/db.sqlite3"]
