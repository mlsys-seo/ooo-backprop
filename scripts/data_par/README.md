# Prepared scripts for Data Parallel Experiments
## 1. Environment Setup

### On-Promise Setup
- Prerequisites: nvidia driver, docker, git

### AWS Setup

- Step 1: Choose an Amazon Machine Images(AMI)
    - Deep Learning AMI (Ubuntu 18.04) Version 56.0 

- Step 2: Choose an Instance Type

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node... | 1 | 4 | `p3.8xlarge` |
| run_4node... | 4 | 16 | `p3.8xlarge` |
| run_8node... | 8 | 32 | `p3.8xlarge` |
| run_12node... | 12 | 64 | `p3.8xlarge` |

- Step 3: Configure Instance
    - All instances have to be in same `Security Group`
    - Open all TCP port inside `Security Group` for the worker communication
