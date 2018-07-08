#!/bin/sh

while true ; do
    # wait until there is an unrated network.
    curl -gs "http://$DB/api/v1/networks?limit=50000" > /tmp/network_list.json

    ANY_UNRATED=`jq 'select(.[].elo == null)' < /tmp/network_list.json`
    NUM_NETWORKS=`jq '. | length' < /tmp/network_list.json`

    if test "$NUM_NETWORKS" -eq 1 ; then
        # set the elo if the singleton network to 0
        ID=`jq -r '.[] | .id' < /tmp/network_list.json`

        if test -n "$ID" ; then
            curl -gs -X PATCH -H "Content-Type: application/json" --data "{\"elo\": 0.0}" "http://$DB/api/v1/networks/$ID"
        fi
    elif test -n "$ANY_UNRATED" ; then
        echo "[`date +%H:%M:%S`] tick"

        # put an instance of the $N latest networks into `dist/`
        rm -rf dist/ && mkdir -p dist/
        curl --compressed -gs "http://$DB/api/v1/networks?limit=$N&full=true" \
            > /tmp/network_list.json

        for i in $(seq 0 $N); do
            NAME=`jq -r ".[$i].name" < /tmp/network_list.json`

            if [ "$NAME" != "null" ]; then
                cp ./dream_go dist/$NAME
                jq -r ".[$i].data" < /tmp/network_list.json > dist/$NAME.json
            fi
        done

        # put an instance of the strongest network into `dist/`
        curl --compressed -gs "http://$DB/api/v1/networks?sort[elo]=desc&limit=1&full=true" \
            > /tmp/best_network.json
        NAME=`jq -r ".[].name" < /tmp/best_network.json`

        cp ./dream_go dist/$NAME
        jq -r ".[].data" < /tmp/best_network.json > dist/$NAME.json

        # run `ringmaster`, and upload the result to the database
        ringmaster ./gomill-playoff.ctl.py -q run
        ./sgf2big.py gomill-playoff.ctl.games/ \
            | ./upload2rest.py --sgf "http://$DB/api/v1/evaluation_games"

        ringmaster ./gomill-playoff.ctl.py reset

        # fetch **all** previous qualifying games and compute the ELO of each
        # network based on them.
        curl --compressed -gs "http://$DB/api/v1/evaluation_games?limit=500000" \
            | jq -r '.[].data' \
            | ./sgf2elo.py > /tmp/network_elo.txt
        curl --compressed -gs "http://$DB/api/v1/networks?limit=500000" \
            > /tmp/network_list.json

        # update the ELO of each network
        while read line; do
            NAME=`echo "$line" | cut -d ':' -f 1`
            ELO=`echo "$line" | rev | cut -f 1 | rev`
            ID=`jq -r ".[] | select(.name == \"$NAME\") | .id" < /tmp/network_list.json`

            if test -n "$ID" ; then
                curl -gs -X PATCH -H "Content-Type: application/json" --data "{\"elo\": $ELO}" "http://$DB/api/v1/networks/$ID"
            fi
        done < /tmp/network_elo.txt

        # wait for at least 10 minutes before running another evaluation trial
        echo "[`date +%H:%M:%S`] sleeping for 10 minutes"

        sleep 600
    else
        echo "[`date +%H:%M:%S`] waiting"

        sleep 300
    fi
done
