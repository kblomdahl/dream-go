#!/bin/sh

while true; do
    # fetch the id of the latest weights from the database, and then extract
    # the following properties:
    #
    # - id
    # - name (for pretty-print)
    # - data
    curl -gs "http://$DB/api/v1/networks?sort[elo]=desc&limit=1&full=true" > network_info.json
    jq -rj ".[0].data" < network_info.json > dream_go.json

    ID=`jq -rj ".[0].id" < network_info.json`
    NAME=`jq -rj ".[0].name" < network_info.json`

    # play some games and then upload them to the database
    echo "[`date +%H:%M:%S`] tick (gen $NAME)"

    ./dream_go $OPTS --policy-play $N | ./sgf2score.py --all | tee self_play.sgf | ./upload2rest.py --sgf "http://$DB/api/v1/games?category=policy_play&network_id=$ID"
    ./dream_go $OPTS --ex-it --extract self_play.sgf | ./upload2rest.py --tfrecord "http://$DB/api/v1/features?network_id=$ID"
done
