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
    echo "::: PULL IMAGE AT "$NODE_HOST" :::"

    ssh -i $SSH_KEY_PATH $SSH_ID@$NODE_HOST \
        sudo docker pull $DOCKER_IMAGE &
done
echo "==============================================================================="
echo ""
echo ""