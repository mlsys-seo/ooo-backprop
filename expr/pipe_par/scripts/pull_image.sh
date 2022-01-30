#!/bin/bash

if [[ $REMOTE_ARG == "preset" ]]
then
    declare -a NODE_HOST_LIST=($NODE_HOSTS_STRING)
else
    source $(dirname $0)/setup.sh
fi



echo ""
echo ""
echo "============================== PULL DOCKER IMAGE =============================="
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    if [[ "$NODE_HOST" != "$MASTER_HOST" ]]
    then
        echo "::: PULL IMAGE AT "$NODE_HOST" :::"

        ssh -i $SSH_KEY_PATH -o "StrictHostKeyChecking no" $SSH_ID@$NODE_HOST \
            "docker pull $DOCKER_IMAGE" &
        echo ""
    fi
done

echo "::: PULL IMAGE AT "$MASTER_HOST" :::"
docker pull $DOCKER_IMAGE
echo ""

echo "==============================================================================="
echo ""
echo ""