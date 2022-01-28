# OutOfOrder_Backprop

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

OutOfOrder Backprop is ~~~

## News
- [OOO-Backprop Paper](link) has been accepted to EuroSys'22!

## Performance

>Single-GPU Training.

>Pipeline-parallel Training.

>Data-parallel Training.

## Quick Start

We provide a [step-by-step tutorial](docs/step-by-step-tutorial.md) for you to run benchmark training tasks. The simplest way to start is to use our [docker images](docker). Refer to [Documentations](docs) for how to [launch distributed jobs](docs/running.md) and more [detailed configurations](docs/env.md). After you can start BytePS, read [best practice](docs/best-practice.md) to get the best performance.

Below, we explain how to install BytePS by yourself. There are two options.

### Start at AWS

>[Single-GPU Training](impl/single-gpu/README.md).

>[Pipeline-Parallel Training](impl/single-gpu/README.md).

>[Data-Parallel Training](impl/single-gpu/README.md).

### Build from source code

You can try out running the experiments via build or pull our docker image.

```
docker pull ~~
```

## Publications
1. [EuroSys] "[PAPER NAME](link)". Hyungjun Oh, --, --, --, Jiwon Seo.
