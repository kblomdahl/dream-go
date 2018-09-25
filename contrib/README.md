# Contributions

This directory contains additional software that is very useful in combination
with _Dream Go_. At the moment the main purpose of this directory is to contain
the training environment.

## Training Environment

The following two docker images are part of the training environment:

* `dream_go/trainer:0.6.0`
* `dream_go/worker:0.6.0`

There is also a server image that is available from a [separate repository](https://github.com/Chicoryn/dream-go-server).

### Setup

```bash
make -C trainer
make -C worker
docker-compose up
```
