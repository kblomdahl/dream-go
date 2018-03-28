#!/bin/sh

tensorboard --logdir models/ &

while true; do
    echo "[`date +%H:%m:%S`] tick"

    # fetch the latest batch of features from the database
    curl -s http://$DB/features/recent/2000000 > /tmp/features.bin

    # if this is the first network then train from scratch, otherwise perform
    # a warm start from the latest model.
    warm_start=`find models/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 ls -1td | head -1`

    if [ "$warm_start" = "." ] || [ "$warm_start" = "models" ] ; then
        python3 -m dream_tf --start --steps 125829120 /tmp/features.bin
    else
        python3 -m dream_tf --start --warm-start "$warm_start" \
            --steps 10485760 /tmp/features.bin
    fi

    # calibrate the scaling factors using a different set of games, and upload
    # the final weights to the database
    curl -s http://$DB/features/recent/20000 > /tmp/calibrate.bin

    python3 -m dream_tf --dump /tmp/calibrate.bin | ./upload2rest.py --bytes -1 http://$DB/weights
done
