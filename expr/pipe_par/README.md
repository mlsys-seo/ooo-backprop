# Implementation OOO of Pipeline-Parallel

- model: BERT
- communication method: `NCCL` for intra-communication, `GRPC` for inter-communication
- `..gpipe.sh`: baseline
- `..modulo.sh`: OOO-Backprop

## AWS common setup
- For multi-node experiments, you must set same `Security Group`.
- `Security Group` must allows all TCP ports within itself.
- For EC2 instanse, Use `Deep Learning AMI (Ubuntu 18.04) Version 56.0` image which already contains everything for experiments(NVIDIA driver, docker, git, etc...)

## Pre-Training Experiments

Launch 8, 16, 24, or 32 AWS instances and edit the script file... Then run the script on the first node (NODE_HOST_LIST[0]).

- `/pretrain`

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node_8gpu... | 1 | 8 | `p3.16xlarge` |
| run_2node_16gpu... | 2 | 16 | `p3.16xlarge` |
| run_2node_16gpu... | 3 | 24 | `p3.16xlarge` |
| run_2node_16gpu... | 4 | 32 | `p3.16xlarge` |



## Fine-Tuning Experiments

- `/finetune`
- Fine-Tuning Task: MRPC of GLUE task

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node_4gpu... | 1 | 4 | `p3.8xlarge` |
| run_4node_4gpu... | 4 | 4 | `p3.2xlarge` |
