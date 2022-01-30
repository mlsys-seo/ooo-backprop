### AWS Configure
We tested the artifact on three AWS instances (i.e., `p3.2xlarge`, `p3.8xlarge`, and `p3.16xlarge`), 
respectively for the single-GPU, data-parallel, and pipeline-parallel training experiments. 
To setup an AWS instance for the experiments, one needs to take the follow steps.

#### Common Setup
- For multi-node experiments, you must set same `Security Group`.
- `Security Group` must allows all TCP ports within itself.
- For EC2 instanse, Use `Deep Learning AMI (Ubuntu 18.04) Version 56.0` image which already contains everything for experiments(NVIDIA driver, docker, git, etc...)

#### EC2 Instance for each experiments

- Single-GPU Training
  - `p3.2xlarge`: 1 X 16G V100 GPUs, up to 10Gbps Ethernet
- Pipeline-Parallel Training
  - `p3.16xlarge`: 8 X 16G V100 GPUs with NVLink, 25Gbps Ethernet
- Data-Parallel Traning
  - `p3.8xlarge`: 4 X 16G V100 GPUs with NVLink, 10Gbps Ethernet
