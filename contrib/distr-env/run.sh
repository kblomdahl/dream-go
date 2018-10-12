#!/bin/sh

while true ; do
    # generate some training data
    for i in `seq 1 25` ; do
        docker-compose run worker
    done

    # create the next network generation
    docker-compose run train
done
