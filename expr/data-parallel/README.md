# Implementation OOO of Data-Parallel on BytePS
## 1. Setup

### AWS instance
```
Step 1: Choose an Amazon Machine Images(AMI)
  Deep Learning AMI (Ubuntu 18.04) Version 56.0 
```

```
Step 2: Choose an Instance Type 
  1 ~ 12 p3.8xlarge Instance
```

```
STep 3: Configure security Group
  Open TCPport 1234 for all instances for the worker communication
```

## 2. Expreriments

```bash
$ git clone https://github.com/mlsys-seo/OutOfOrder_Backprop.git
$ cd scripts/data-parallel
$ ./run_gpu4_resnet50.sh
```
