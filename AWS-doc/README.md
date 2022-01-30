# AWS Configure
We tested the artifact on three AWS instances (i.e., `p3.2xlarge`, `p3.8xlarge`, and `p3.16xlarge`), 
respectively for the single-GPU, data-parallel, and pipeline-parallel training experiments. 
To setup an AWS instance for the experiments, oneneeds to take the follow steps.

## Single-GPU Training
  `p3.2xlarge`: 1 X `16G V100` GPUs, up to 10Gbps Ethernet

## Data-Parallel Traning
  `p3.8xlarge`: 4 X `16G V100` GPUs with `NVLink`, 10Gbps Ethernet

## Pipeline-Parallel Training
  `p3.16xlarge`: 8 X `16G V100` GPUs with `NVLink`, 25Gbps Ethernet
