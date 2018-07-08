#!/bin/sh

export TF_CPP_MIN_LOG_LEVEL=2

tensorboard --logdir models/ 2> /dev/null &

while true; do
    # wait until:
    #
    # - all networks has been rated
    # - the best rated network has at least 40,000 played games
    #
    NUM_FEATURES=`curl -sg "http://$DB/api/v1/networks?sort[elo]=desc&limit=1" | jq -r '.[].number_of_features'`
    ANY_UNRATED=`curl -sg "http://$DB/api/v1/networks?limit=50000" | jq 'select(.[].elo == null)'`

    if test $NUM_FEATURES -lt 40000 -o -n "$ANY_UNRATED" ; then
        echo "[`date +%H:%M:%S`] waiting ($NUM_FEATURES)"

        sleep 300
    else
        echo "[`date +%H:%M:%S`] tick"

        # fetch the latest batch of features from the database
        LIMIT=500000

        curl --compressed -sg "http://$DB/api/v1/features?limit=$LIMIT" | jq -r '.[].data' | base64 --decode > /tmp/features.tfrecord

        # if this is the first network then train from scratch, otherwise perform
        # a warm start from the latest model.
        warm_start=`find models/ -mindepth 1 -maxdepth 1 -type d -print0 | xargs -0 ls -1td | head -1`
        name=`petname -w3 -s_`

        if [ "$warm_start" = "." ] || [ "$warm_start" = "models" ] ; then
            python3 -m dream_tf --name "$name" --start /tmp/features.tfrecord
        else
            python3 -m dream_tf --name "$name" --start --warm-start "$warm_start" --steps $((20 * $LIMIT)) /tmp/features.tfrecord
        fi

        # upload the final weights to the database
        python3 -m dream_tf --dump | ./upload2rest.py --bytes -1 "http://$DB/api/v1/networks?name=$name"
    fi
done
