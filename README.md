# OutOfOrder_Backprop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

OutOfOrder Backprop is ~~~

the code of the ~~~ is in ```tensorflow/```.  

the code of the ~~~ is in ```byteps/```.  

The experiments presented in the paper, including replication instructions, are in ```experiments/```.  


## News
- [OOO-Backprop Paper](link) has been accepted to EuroSys'22!

## Performance
OOO BackProp is evaluated with twelve neural network and five public datasets. Compared to the respective state of the art training systems, It improves the training throughput by 1.03-1.58x for single-GPU training, by 1.10–1.27× for data-parallel training, and by 1.41–1.99× for pipeline-parallel training.


>Single-GPU Training.

![single](https://user-images.githubusercontent.com/78071764/151532657-bb4a35c3-83bc-49a4-8792-2a4b3277dc7d.png)


>Pipeline-parallel Training.

![pipeline](https://user-images.githubusercontent.com/78071764/151532720-0c64410a-317d-4c6b-a4b4-8b96c622aae1.png)

>Data-parallel Training.

![datap](https://user-images.githubusercontent.com/78071764/151532987-d56e3311-407d-406e-b389-ab811267eda9.png)


## Quick Start

Below, we introduce two options how to reproduce our experiments.

### Start at AWS

#### [Single-GPU Training](impl/single_gpu/README.md)

#### [Pipeline-Parallel Training](impl/pipeline_parallel/README.md)

#### [Data-Parallel Training](impl/data_parallel/README.md)

### Build from source code

You can try out running the experiments via build or pull our docker image.

```
git clone --recursive https://github.com/mlsys-seo/OutOfOrder_Backprop.git
cd tensorflow
ooo_tensorflow_build.sh
```

## Publications
1. [EuroSys] "[PAPER NAME](link)". Hyungjun Oh, --, --, --, Jiwon Seo.
