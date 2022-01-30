# Prepared scripts for Data Parallel Experiments

- model: `ResNet-50` and `ResNet-101`
- communication method: `NCCL` for intra-communication, `GRPC` for inter-communication
- `..BytePS.sh`: baseline
- `..OOO-Backprop.sh`: OOO-Backprop

## AWS common setup
- For multi-node experiments, you must set same `Security Group`.
- `Security Group` must allows all TCP ports within itself.
- For EC2 instanse, Use `Deep Learning AMI (Ubuntu 18.04) Version 56.0` image which already contains everything for experiments(NVIDIA driver, docker, git, etc...)

## Expreiments

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node... | 1 | 4 | `p3.8xlarge` |
| run_4node... | 4 | 16 | `p3.8xlarge` |
| run_8node... | 8 | 32 | `p3.8xlarge` |
| run_12node... | 12 | 64 | `p3.8xlarge` |
