#!/bin/sh

while true; do
    # fetch the id of the latest weights from the database, and then get the
    # weights themselves. We do it this way so that we can track what generation
    # of weights each self-play game was generated from.
    GEN=`curl -s http://$DB/weights/recent/1/rowid`
    curl -s http://$DB/weights/$GEN > dream_go.json

    # play some games and then upload them to the database
    NOW=`date +%H:%m:%S`
    echo "[$NOW] tick (gen $GEN)"

    ./dream_go $OPTS --policy-play $N | ./sgf2score.py --all | tee self_play.sgf | ./upload2rest.py --sgf "http://$DB/policy_play?generation=$GEN"
    ./dream_go $OPTS --ex-it --extract self_play.sgf | ./upload2rest.py --bytes 1987 "http://$DB/features?generation=$GEN"
done
