# OutOfOrder Backprop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

OutOfOrder Backprop is an effective scheduling technique for neural network training. By exploiting the dependencies of gradient computations, ooo backprop enables to reorder their executions to make the most of the GPU resources. We show that the GPU utilization in single and multi-GPU training can be commonly improve by applying ooo backprop and prioritizing critical operations. 
We propose three scheduling algorithms based on ooo backprop. For single-GPU training, we schedule with multi-stream ooo computation to mask the kernel launch overhead. In data-parallel training, we reorder the gradient computations to maximize the overlapping of computation and parameter communication; in pipeline-parallel training, we prioritize critical gradient computations to reduce the pipeline stalls.

```tensorflow/``` Source code of TensorFlow (v2.4) modifiedto (optionally) run with ooo backprop.

```byteps/``` Source code of BytePS (v0.2.5) modified to (op-tionally) run with ooo backprop.

```expr/``` Python scripts for defining and training the eval-uated models. Three sub-directories contain the code forthe three sets of experiments.

```AWS-doc/``` Documentation for setting AWS instances for the experiments.

```scripts/``` Bash scripts for running all the experiments.


## Performance
OOO BackProp is evaluated with twelve neural network and five public datasets. Compared to the respective state of the art training systems, It improves the training throughput by 1.03-1.58x for single-GPU training, by 1.10–1.27× for data-parallel training, and by 1.41–1.99× for pipeline-parallel training.


>Single-GPU Training.

![single](https://user-images.githubusercontent.com/78071764/151532657-bb4a35c3-83bc-49a4-8792-2a4b3277dc7d.png)


>Pipeline-parallel Training.

![pipeline](https://user-images.githubusercontent.com/78071764/151532720-0c64410a-317d-4c6b-a4b4-8b96c622aae1.png)

>Data-parallel Training.

![datap](https://user-images.githubusercontent.com/78071764/151532987-d56e3311-407d-406e-b389-ab811267eda9.png)


## Quick Start

### AWS Configure
We tested the artifact on three AWS instances (i.e., `p3.2xlarge`, `p3.8xlarge`, and `p3.16xlarge`), 
respectively for the single-GPU, data-parallel, and pipeline-parallel training experiments. 
To setup an AWS instance for the experiments, one needs to take the follow steps.

#### Common Setup
- For multi-node experiments, you must set same `VPC`, `Subnet` and `Security Group`.
- `Security Group` must allows all TCP ports within itself.
- We strongly recommend `Deep Learning AMI (Ubuntu 18.04) Version 56.0` ami which already containes everything for experiments(NVIDIA driver, docker, git, etc...)

#### EC2 Instance for each experiments

1. Single-GPU Training
  - `p3.2xlarge`: 1 X 16G V100 GPUs, up to 10Gbps Ethernet
1. Pipeline-Parallel Training
  - `p3.16xlarge`: 8 X 16G V100 GPUs with NVLink, 25Gbps Ethernet
1. Data-Parallel Traning
  - `p3.8xlarge`: 4 X 16G V100 GPUs with NVLink, 10Gbps Ethernet

### [Single-GPU Training](expr/single_gpu/README.md)

### [Pipeline-Parallel Training](expr/pipe_par/README.md)

### [Data-Parallel Training](expr/data-par/README.md)

