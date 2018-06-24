#!/bin/sh

export DB="localhost:5000"
export N=2

while true ; do
    echo "[`date +%H:%M:%S`] tick"

    # put an instance of the $N latest networks into `dist/`
    rm -rf dist/ && mkdir -p dist/
    curl -s "http://$DB/api/v1/networks?limit=$N&full=true" \
        > /tmp/network_list.json

    for i in $(seq 0 $N); do
        NAME=`jq -r ".[$i].name" < /tmp/network_list.json`

        if [ "$NAME" != "null" ]; then
            cp ./dream_go dist/$NAME
            jq -r ".[$i].data" < /tmp/network_list.json > dist/$NAME.json
        fi
    done

    # run `ringmaster`, and upload the result to the database
    ringmaster ./gomill-playoff.ctl.py -q run
    ./sgf2big.py gomill-playoff.ctl.games/ \
        | ./upload2rest.py --sgf "http://$DB/api/v1/games?category=evaluation&network_id=1"

    ringmaster ./gomill-playoff.ctl.py reset

    # fetch **all** previous qualifying games and compute the ELO of each
    # network based on them.
    curl -gs "http://$DB/api/v1/games?filter[category]=evaluation&limit=500000" \
        | jq -r '.[].data' \
        | ./sgf2elo.py > /tmp/network_elo.txt
    curl -s "http://$DB/api/v1/networks?limit=500000" \
        > /tmp/network_list.json

    # update the ELO of each network
    while read line; do
        NAME=`echo "$line" | cut -d ':' -f 1`
        ELO=`echo "$line" | rev | cut -f 1 | rev`
        ID=`jq -r ".[] | select(.name == \"$NAME\") | .id" < /tmp/network_list.json`

        curl -s -X PATCH -H "Content-Type: application/json" --data "{\"elo\": $ELO}" "http://$DB/api/v1/networks/$ID"
    done < /tmp/network_elo.txt

    # wait for three hours before running another evaluation trial
    echo "[`date +%H:%M:%S`] sleeping for three hours"

    sleep 10800
done
