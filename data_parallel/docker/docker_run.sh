docker run -d -it \
    -v ${PWD}:/workspace/bdsl \
    --name sk-ooo-byteps \
    --privileged --ipc=host --net=host \
    --restart=always \
    --gpus=all \
    195164261969.dkr.ecr.us-west-2.amazonaws.com/ooo_byteps:tf2 bash

docker run -d -it \
    -v ${PWD}:/workspace/bdsl \
    --name sk-ooo-byteps \
    --privileged --ipc=host --net=host \
    --restart=always \
    --gpus=all \
    mlsys.duckdns.org:9999/ooo-backprop-byteps bash
    