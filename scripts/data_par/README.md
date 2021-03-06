# Prepared scripts for Data Parallel Experiments

- Implemantation code is [here](../../expr/data_par)

## 1. Environment Setup

### On-Promise Setup
- Prerequisites: nvidia driver, docker, git

### AWS Setup

- Step 1: Choose an Amazon Machine Images(AMI)
    - Deep Learning AMI (Ubuntu 18.04) Version 56.0 


- Step 2: Choose an Instance Type
    - For 4 gpus / node: `p3.8xlarge` instance


- Step 3: Configure Instance
    - All instances have to be in same `Security Group`
    - Open all TCP port inside `Security Group` for the worker communication

## 2. Run the Experiments

```bash
$ git clone https://github.com/mlsys-seo/ooo-backprop
$ cd ooo-backprop/scripts/data_par
$ # Edit the following script for MASTER_HOST, NODE_HOST_LIST[], SSH_KEY_PATH, SSH_ID
$ ./run_1node_4gpu_resnet50_BytePS.sh
```
Note that you need to git-clone the repository and run the script only in the master node (that is, MASTER_HOST in the scripts which should be the same as NODE_HOST_LIST[0]).
