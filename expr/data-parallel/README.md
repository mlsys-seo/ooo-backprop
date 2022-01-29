# Implementation OOO of Data-Parallel on BytePS

## 1. Setup

### Build from Dockerfile at local
```bash
$ docker build -t ooo-backprop/data-parallel -f docker/Dockerfile .
```

### Pull Docker Image
```bash
$ docker pull ooo-backprop/data-parallel:latest
```

## 1. Setup
1. Launch 1 ~ 12 AWS p3.8xlarge instances
1. Copy the downloaded images to the AWS instances
1. Run the following commands in all the instances

## 2. Run on AWS
- Environment setting
  - Amazon AWS p3.8xlarge


## 2. Expreriments


```bash
$ ./scripts/remote/run.sh
```
