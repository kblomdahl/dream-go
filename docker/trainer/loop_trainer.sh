#!/bin/sh

tensorboard --logdir logs/ &

while true; do
    now=`date +%H:%m:%S`

    echo "[$now] tick"

    # fetch the latest batch of features from the database
    curl -s http://$DB/features/recent/1000000 > features.bin

    rm -f models/*
    python3 bootstrap.py features.bin

    # calibrate the scaling factors using a different set of games, and upload
    # the final weights to the database
    curl -s http://$DB/features/recent/20000 > calibrate.bin

    python3 bootstrap.py --dump calibrate.bin | ./upload2rest.py --bytes -1 http://$DB/weights
done
