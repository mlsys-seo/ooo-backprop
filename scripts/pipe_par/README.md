# Prepared scripts for Data Parallel Experiments

- `model`: BERT
- `communication method`: `NCCL` for intra-communication, `GRPC` for inter-communication
- `..gpipe.sh`: baseline
- `..modulo.sh`: OOO-Backprop

## Pre-Training Experiments

- `/pretrain`

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node_8gpu... | 1 | 8 | `p3.16xlarge` |
| run_2node_16gpu... | 2 | 16 | `p3.16xlarge` |
| run_2node_16gpu... | 3 | 24 | `p3.16xlarge` |
| run_2node_16gpu... | 4 | 32 | `p3.16xlarge` |



## Fine-Tuning Experiments

- `/finetune`
- Fine-Tuning Task: MRPC of GLUE task

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node_4gpu... | 1 | 4 | `p3.8xlarge` |
| run_4node_4gpu... | 4 | 4 | `p3.2xlarge` |
