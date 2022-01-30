# Implementation OOO of Pieline-Parallel
## 1. Setup

### AWS instance
```
Step 1: Choose an Amazon Machine Images(AMI)
  Deep Learning AMI (Ubuntu 18.04) Version 56.0 
```

```
Step 2: Choose an Instance Type 
  1 ~ 4 p3.16xlarge Instance
```

```
STep 3: Configure security Group
  Open all TCPport for all instances for the worker communication
```

## 2. Expreriments 

```bash
$ git clone https://github.com/mlsys-seo/OutOfOrder_Backprop.git
$ cd scripts/pipe_par/finetune
$ ./run_1node_4gpu_finetune_modulo.sh
```
