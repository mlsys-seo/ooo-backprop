#!/bin/bash

echo ""
echo ""
echo "============================== PULL DOCKER IMAGE =============================="
for node_idx in "${!NODE_HOST_LIST[@]}"
do
    NODE_HOST="${NODE_HOST_LIST[$node_idx]}"
    echo "::: PULL IMAGE AT "$NODE_HOST" :::"

    ssh -i $SSH_KEY_PATH -f $SSH_ID@$NODE_HOST \
        docker pull $DOCKER_IMAGE &
done

echo "==============================================================================="
echo ""
echo ""