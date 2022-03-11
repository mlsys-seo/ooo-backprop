# Out-Of-Order Backprop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Out-Of-Order(OOO) Backprop is an effective scheduling technique for neural network training. By exploiting the dependencies of gradient computations, ooo backprop enables to reorder their executions to make the most of the GPU resources. We show that the GPU utilization in single and multi-GPU training can be commonly improve by applying ooo backprop and prioritizing critical operations. 
We propose three scheduling algorithms based on ooo backprop. For single-GPU training, we schedule with multi-stream ooo computation to mask the kernel launch overhead. In data-parallel training, we reorder the gradient computations to maximize the overlapping of computation and parameter communication; in pipeline-parallel training, we prioritize critical gradient computations to reduce the pipeline stalls.

Repository Structure: 

```AWS-doc/``` Simple tips to set up AWS Instances for reproducing experiments.

```tensorflow/``` Source code of TensorFlow (v2.4) modified to (optionally) run with ooo backprop.

```byteps/``` Source code of BytePS (v0.2.5) modified to (optionally) run with ooo backprop.

```expr/``` Python scripts for defining and training the evaluated models. Three sub-directories contain the code forthe three sets of experiments.

```scripts/``` Bash scripts for running all the experiments.

## Quickstart
- [about AWS setup](AWS-doc)
- [Prepared Scripts for Single-GPU Training](scripts/single_gpu/)
- [Prepared Scripts for Pipeline-Parallel Training](scripts/pipe_par/)
- [Prepared Scripts for Data-Parallel Training](scripts/data_par/)

## Install Guide

### Tensorflow Install Guide
### Prerequisites
- Bazel 3.1.0
- CUDA 11.0
- cuDNN 8.0.4
```bash
git clone https://github.com/mlsys-seo/ooo-backprop.git
cd tensorflow
./configure
./build.sh
./install.sh
```
If you want to get more information about installation, please go to [Tensorflow](https://www.tensorflow.org/install/source?hl=ko).

### BytePS Install Guide
```bash
git clone https://github.com/mlsys-seo/ooo-backprop.git
cd byteps
python3 setup.py install
```
If you want to get more information about installation, please go to [BytePS repository](https://github.com/bytedance/byteps).


## Repository Location for each set of experiments
To run more experiments (other than our prepared ones), see the following locatons.

- [Single-GPU Training](expr/single_gpu/)
- [Pipeline-Parallel Training](expr/pipe_par/)
- [Data-Parallel Training](expr/data_par/)
 
## Performance
OOO BackProp is evaluated with twelve neural network and five public datasets. The following is a subset of the evaluation results for single-GPU, data-parallel, and pipeline-parallel training experiments.


### Single-GPU Training.

![single](https://user-images.githubusercontent.com/78071764/151532657-bb4a35c3-83bc-49a4-8792-2a4b3277dc7d.png)


### Pipeline-parallel Training.

![pipeline](https://user-images.githubusercontent.com/78071764/151532720-0c64410a-317d-4c6b-a4b4-8b96c622aae1.png)

### Data-parallel Training.

![datap](https://user-images.githubusercontent.com/78071764/151532987-d56e3311-407d-406e-b389-ab811267eda9.png)
