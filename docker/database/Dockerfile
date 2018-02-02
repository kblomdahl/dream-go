FROM ubuntu
RUN apt-get update -y && apt-get install -y python3
EXPOSE 8080

RUN mkdir -p /app/
COPY staging/database.py /app/database.py

WORKDIR /app
ENTRYPOINT ["/usr/bin/python3", "database.py", "/mnt/dream_db/db.sqlite3"]
