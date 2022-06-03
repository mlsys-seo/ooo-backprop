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

```Modifications.md``` The descriptions of our code modifications to implement ooo backprop in TensorFlow. 

## Quickstart
You can simply run the experiments in our prepared Docker containers by following the instructions in the links below.
To reproduce the experiments in the same environment in our paper, you need to setup the AWS instances as described in the following link.

- [about AWS setup](AWS-doc)
- [Prepared Scripts for Single-GPU Training](scripts/single_gpu/)
- [Prepared Scripts for Pipeline-Parallel Training](scripts/pipe_par/)
- [Prepared Scripts for Data-Parallel Training](scripts/data_par/)

## Requirements
We have prepared and tested the artifact in the following
hardware/software settings

- A Linux machine with kernel 5.4.0-1063-aws (Ubuntu 18.04).
- Intel Xeon E5-2686 v4 (2.3 GHz) and NVIDIA V100 GPU.
- CUDA v11.0 and GPU driver version 450.142.00
- A GPU cluster consisting of V100 GPUs with NVLink inter-GPU interconnect and 10 or 25Gb inter-node interconnect (i.e., AWS p3.8xlarge or p3.16xlarge).
- For simply running the experiments, any Linux machine that runs TensorFlow/BytePS is sufficient.

Although CPU or Linux kernel should not affect our optimizations, we suggest to use the above settings to reproduce our experimental results.

## Install Guide
If you want to run the experiments in our prepared Docker containers, follow the links in Quickstart above.
If you prefer to run the experiments in your own Linux machine, you need to compile TensorFlow and BytePS as following.

### Tensorflow Install Guide
### Prerequisites
- Bazel 3.1.0

```bash
export WHEEL_DIR=/your/tensorflow/wheel/path/

git clone https://github.com/mlsys-seo/ooo-backprop.git
cd tensorflow

./configure
bazel build --config=opt --config=cuda --strip=never //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package ${WHEEL_DIR}

# If there is tensorflow already, remove the package and install the newly created package.
# You need to find out the "Two-Digits" by checking the created directory.
pip uninstall -y tensorflow
pip install ${WHEEL_DIR}/tensorflow-2.4.0-cp"Two-Digits"-cp"Two-Digits"m-linux_x86_64.whl

# There may be an issue about protocol buffer package version. Downgrade the package to the version 3.10.0
pip uninstall -y protobuf
pip install protobuf==3.10.0
```
For Tensorflow package dependencies, please go to [Tensorflow](https://www.tensorflow.org/install/source?hl=ko).

### BytePS Install Guide
```bash
git clone https://github.com/mlsys-seo/ooo-backprop.git
cd byteps
python3 setup.py install
```
For package dependencies and more detailed installation information, please go to [BytePS repository](https://github.com/bytedance/byteps).

## Running Scripts with Compiled TF/BytePS

Rather than using the script files in scripts/{single\_gpu, data\_par, pipe\_par}/\*.sh to run the experiments, 
you need to use the scripts in the following.

* Single-GPU Traing Expr: expr/single\_gpu/scripts/run_\*.sh
* Data-Parallel Training Expr: expr/data\_par/code/run\_node\_resnet.sh
* Pipeline-Parallel Training Expr: expr/pipe\_par/code/run\_node.sh

To run the scripts, you need to pass arguments to the above scripts; the arguemnts are described at the beginning of the scripts.


## Training with Your Data Using OOO BackProp

If you want to use the provided code to train with your data, you can replace dummy_X and dummy_Y variables in expr/{single_gpu, data_par, pipe_par}/code/\*.py to be placeholders of your loaded data. You may need to optimize the code for loading and preparing your data for training to take full advantage of ooo backprop.

## Repository Location for Each Set of Experiments
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

## Publication

Hyungjun Oh, Junyeol Lee, Hyeongju Kim, and Jiwon Seo, [Out-Of-Order BackProp:
An Effective Scheduling Technique for Deep Learning](https://dl.acm.org/doi/pdf/10.1145/3492321.3519563) in European Conference on Computer Systems (EuroSys), 2022

### Citation

If you use this work, please cite our paper published in European Conference on Computer Systems (EuroSys), 2022

```bibtex
@inproceedings{oh2022out,
  title={Out-of-order backprop: an effective scheduling technique for deep learning},
  author={Oh, Hyungjun and Lee, Junyeol and Kim, Hyeongju and Seo, Jiwon},
  booktitle={Proceedings of the Seventeenth European Conference on Computer Systems},
  pages={435--452},
  year={2022}
}
```
