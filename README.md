# Out-Of-Order Backprop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Out-Of-Order(OOO) Backprop is an effective scheduling technique for neural network training. By exploiting the dependencies of gradient computations, ooo backprop enables to reorder their executions to make the most of the GPU resources. We show that the GPU utilization in single and multi-GPU training can be commonly improve by applying ooo backprop and prioritizing critical operations. 
We propose three scheduling algorithms based on ooo backprop. For single-GPU training, we schedule with multi-stream ooo computation to mask the kernel launch overhead. In data-parallel training, we reorder the gradient computations to maximize the overlapping of computation and parameter communication; in pipeline-parallel training, we prioritize critical gradient computations to reduce the pipeline stalls.

```tensorflow/``` Source code of TensorFlow (v2.4) modified to (optionally) run with ooo backprop.

```byteps/``` Source code of BytePS (v0.2.5) modified to (optionally) run with ooo backprop.

```expr/``` Python scripts for defining and training the eval-uated models. Three sub-directories contain the code forthe three sets of experiments.

```scripts/``` Bash scripts for running all the experiments.


## Performance
OOO BackProp is evaluated with twelve neural network and five public datasets. Compared to the respective state of the art training systems, It improves the training throughput by 1.03-1.58x for single-GPU training, by 1.10–1.27× for data-parallel training, and by 1.41–1.99× for pipeline-parallel training.


- Single-GPU Training.

![single](https://user-images.githubusercontent.com/78071764/151532657-bb4a35c3-83bc-49a4-8792-2a4b3277dc7d.png)


- Pipeline-parallel Training.

![pipeline](https://user-images.githubusercontent.com/78071764/151532720-0c64410a-317d-4c6b-a4b4-8b96c622aae1.png)

- Data-parallel Training.

![datap](https://user-images.githubusercontent.com/78071764/151532987-d56e3311-407d-406e-b389-ab811267eda9.png)


## Quickstart
- [about AWS setup](AWS-doc)
- [Single-GPU Training](scripts/single_gpu/)
- [Pipeline-Parallel Training](scripts/pipe_par/)
- [Data-Parallel Training](scripts/data_par/)

## Implemantaion
- [Single-GPU Training](expr/single_gpu/)
- [Pipeline-Parallel Training](expr/pipe_par/)
- [Data-Parallel Training](expr/data_par/)
