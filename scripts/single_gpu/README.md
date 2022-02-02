# Prepared scripts for Single GPU Experiments

## 1. Environment Setup

### On-Promise Setup
- Prerequisites: nvidia driver, docker, git

### AWS Setup

- Step 1: Choose an Amazon Machine Images(AMI)
    - Deep Learning AMI (Ubuntu 18.04) Version 56.0 

- Step 2: Choose an Instance Type
    - Node: `p3.2xlarge` instance

## 2. Run the expriments

```bash
git clone https://github.com/mlsys-seo/ooo-backprop.git
cd ooo-backprop

# To run OOO-BackProp single GPU experiment
$ ./scripts/single_gpu/single_gpu_densenet_k24_b32_ooo.sh
