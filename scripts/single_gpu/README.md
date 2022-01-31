# Prepared scripts for Single GPU Experiments

- Implemantation code is on [here](../../expr/single_gpu)
- model: DenseNet 

## AWS common setup
- For multi-node experiments, you must set same `Security Group`.
- `Security Group` must allows all TCP ports within itself.
- For EC2 instanse, Use `Deep Learning AMI (Ubuntu 18.04) Version 56.0` image which already contains everything for experiments(NVIDIA driver, docker, git, etc...)

## Experiments

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| single_gpu_densenet_k12_b32_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k12_b32_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a0.25_b32_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a0.25_b32_ooo.sh | 1 | 1 | `p3.2xlarge` |
