docker run -d -it \
    -v ${PWD}:/workspace/bdsl \
    --name sk-test \
    --privileged --ipc=host --net=host \
    --restart=always \
    --gpus=all \
    nvidia/cuda:11.1.1-cudnn8-devel-ubuntu18.04