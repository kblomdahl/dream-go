#!/bin/sh

while true ; do
    # generate some training data
    for i in `seq 1 25` ; do
        docker-compose run --rm worker
    done

    # create the next network generation
    docker-compose run --rm train
done
