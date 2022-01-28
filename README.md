# OutOfOrder_Backprop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

OutOfOrder Backprop is ~~~

## News
- [OOO-Backprop Paper](link) has been accepted to EuroSys'22!

## Performance

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
docker pull ~~
```

## Publications
1. [EuroSys] "[PAPER NAME](link)". Hyungjun Oh, --, --, --, Jiwon Seo.
