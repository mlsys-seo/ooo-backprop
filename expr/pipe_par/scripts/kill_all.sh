#!/bin/bash

if [[ $REMOTE_ARG == "preset" ]]
then
    declare -a NODE_HOST_LIST=($NODE_HOSTS_STRING)
else
    source $(dirname $0)/setup.sh
fi


echo ""
echo ""
echo "=============================== Kill Containers ==============================="

for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    if [[ "$NODE_HOST" != "$MASTER_HOST" ]]
    then
        echo "::: KILL CONTAINER AT "$NODE_HOST" :::"
        ssh -i $SSH_KEY_PATH $SSH_ID@$NODE_HOST \
            docker kill ooo-pipe-$node_idx
    fi
done

echo "::: PULL IMAGE AT "$MASTER_HOST" :::"
docker kill ooo-pipe-0
echo ""

echo "==============================================================================="
echo ""
echo ""