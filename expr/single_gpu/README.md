# Implementation OOO on Single-GPU

## 1. Environment Setup

### On-Promise Setup
- Prerequisites: nvidia driver, docker, git

### AWS Setup

- Step 1: Choose an Amazon Machine Images(AMI)
    - Deep Learning AMI (Ubuntu 18.04) Version 56.0 


- Step 2: Choose an Instance Type
    - For 1 gpu / node: `p3.2xlarge` instance

- Step 3: Configure Instance
    - All instances have to be in same `Security Group`
    - Open all TCP port inside `Security Group` for the worker communication

## 2. Expriments


```bash
$ ./scripts/remote/run.sh
```
