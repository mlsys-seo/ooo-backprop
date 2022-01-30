# Implementation OOO of Pieline-Parallel

## 1. Environment Setup
### AWS Setup

1. Step 1: Choose an Amazon Machine Images(AMI)
    - Deep Learning AMI (Ubuntu 18.04) Version 56.0 

2. Step 2: Choose an Instance Type
    - For 1 gpu / node: `p3.2xlarge` instance
    - For 4 gpus / node: `p3.8xlarge` instance
    - For 8 gpus / node: `p3.16xlarge` instance

3. Step 3: Configure Instance
    - All instances have to be in same `Security Group`
    - Open all TCP port inside `Security Group` for the worker communication

## 2. Expreriments 

```bash
$ git clone https://github.com/mlsys-seo/OutOfOrder_Backprop.git
$ cd OutOfOrder_Backprop/scripts/pipe_par/finetune
$ # Edit MASTER_HOST, NODE_HOST_LIST[], SSH_KEY_PATH, SSH_ID
$ ./run_1node_4gpu_finetune_gpipe.sh
```
