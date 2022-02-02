# Implementation OOO on Single-GPU

- model: DenseNet, MobileNet

## AWS common setup
- For EC2 instanse, Use `Deep Learning AMI (Ubuntu 18.04) Version 56.0` image which already contains everything for experiments(NVIDIA driver, docker, git, etc...)

## Experiments

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| single_gpu_densenet_k24_b32_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k24_b32_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k32_b32_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k32_b32_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k24_b64_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k24_b64_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k32_b64_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_densenet_k32_b64_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a0.5_b32_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a0.5_b32_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a1.0_b32_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a1.0_b32_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a0.5_b64_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a0.5_b64_ooo.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a1.0_b64_base.sh | 1 | 1 | `p3.2xlarge` |
| single_gpu_mobilenet_v3_a1.0_b64_ooo.sh | 1 | 1 | `p3.2xlarge` |
