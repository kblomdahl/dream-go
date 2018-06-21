#!/bin/sh

tensorboard --logdir models/ &

while true; do
    echo "[`date +%H:%m:%S`] tick"

    # fetch the latest batch of features from the database
    curl -s http://$DB/api/v1/features?limit=500000 | jq -r '.[].data' | base64 --decode > /tmp/features.bin

    # if this is the first network then train from scratch, otherwise perform
    # a warm start from the latest model.
    warm_start=`find models/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 ls -1td | head -1`

    if [ "$warm_start" = "." ] || [ "$warm_start" = "models" ] ; then
        python3 -m dream_tf --start /tmp/features.bin
    else
        python3 -m dream_tf --start --warm-start "$warm_start" /tmp/features.bin
    fi

    # upload the final weights to the database
    python3 -m dream_tf --dump | ./upload2rest.py --bytes -1 "http://$DB/api/v1/networks?name=`petname -w3 -s_`"
done
