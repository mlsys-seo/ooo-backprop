# Prepared scripts for Data Parallel Experiments

- For multi-node experiments, you must set same `Security Group`.
- `Security Group` must allows all TCP ports within itself.
- For EC2 instanse, Use `Deep Learning AMI (Ubuntu 18.04) Version 56.0` image which already contains everything for experiments(NVIDIA driver, docker, git, etc...)

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node... | 1 | 4 | `p3.8xlarge` |
| run_4node... | 4 | 16 | `p3.8xlarge` |
| run_8node... | 8 | 32 | `p3.8xlarge` |
| run_12node... | 12 | 64 | `p3.8xlarge` |
