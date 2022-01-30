# Prepared scripts for Data Parallel Experiments

- `model`: BERT
- `communication method`: `NCCL` for intra-communication, `GRPC` for inter-communication
- `..gpipe.sh`: baseline
- `..modulo.sh`: OOO-Backprop

## Pre-Training Experiments

- `/pretrain`

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node... | 1 | 4 | `p3.8xlarge` |
| run_4node... | 4 | 16 | `p3.8xlarge` |
| run_8node... | 8 | 32 | `p3.8xlarge` |
| run_12node... | 12 | 64 | `p3.8xlarge` |

## Fine-Tuning Experiments

- `/finetune`
- Fine-Tuning Task: MRPC of GLUE task

| script | number of nodes | number of GPUs | AWS instance |
|:---:|:---:|:---:|:---:|
| run_1node... | 1 | 4 | `p3.8xlarge` |
| run_4node... | 4 | 16 | `p3.8xlarge` |
| run_8node... | 8 | 32 | `p3.8xlarge` |
| run_12node... | 12 | 64 | `p3.8xlarge` |
