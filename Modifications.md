# Overview

Our optimized scheduling algorithms for data-parallel and pipeline-parallel training are implemented entirely in Python, which is described at the botton of this page.
For the single-GPU training, our optimization, that is, multi-stream ooo computation, requires modifications in TensorFlow, which we first describe below.

## Changes in TensorFlow for single-GPU Training

The source code modification in TensorFlow XLA is for multi-stream out-of-order computation (Section 4.1) and pre-compiled kernel issue (Section 4.2) for single-GPU training. The following (A–D) are the locations and descriptions of our modifications.

### A. Stream allocation

 * Source code: tensorflow/tensorflow/compiler/xla/service/gpu/stream_assignment.cc (line: 81–96)

* Github link: [stream_assignment.cc](https://github.com/mlsys-seo/ooo-backprop/blob/b5a535559267db1527bd7b413195d22d6faed3a6/tensorflow/tensorflow/compiler/xla/service/gpu/stream_assignment.cc#L81)

We modified the function AssignStreamToHlo() to assign the streams to kernels as they are annotated in our Python-level scheduler, the results of which are in {densenet,mobilenet_v3}_schedule_map.py.



### B. Enforcing the kernel schedule for main- and sub-streams
* Source code: tensorflow/tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc (line: 470–472)

* Github link: [gpu_hlo_schedule.cc](https://github.com/mlsys-seo/ooo-backprop/blob/b5a535559267db1527bd7b413195d22d6faed3a6/tensorflow/tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc#L471)

We modified the function GpuHloSchedule::Build() to invoke our implemented function MakeOOOLaunchOrder(), which is in line 308. The function enforces the execution schedule for the kernels in main-stream and sub-stream. Note that the kernel scheduling is already determined and stored in {densenet,mobilenet_v3}_schedule_map.py and here we simply enforce the determined schedule.




### C. CUDA Graph Capturing
* Source code: tensorflow/tensorflow/compiler/xla/service/gpu/gpu_executable.cc (line: 927–947)

* Github link: [gpu_executable.cc](https://github.com/mlsys-seo/ooo-backprop/blob/9297229f2d8ef1ebf28a507ab38d16e28639f32d/tensorflow/tensorflow/compiler/xla/service/gpu/gpu_executable.cc#L927)


We modified the function GpuExecutable::ExecuteAsyncOnStream() to invoke our implemented functions, i.e., RewireWeightGradInputs() and ExecuteThunksAndGraphCapture(). The function ExecuteThunksAndGraphCapture() (in line 161) is modified from the function ExecuteThunks() (in line 379) and it has the code for CUDA graph capturing (line 220, 323–342). 

The function RewireWeightGradInputs() (in line 698) is applied if a weight gradient computation is scheduled to overlap with a forward computation in the next iteration. Because XLA performs the scheduling within a single iteration, if we delay a weight grad computation to the next iteration, its memory allocation logic (for multi-stream) does not work. To re-use the existing memory allocation logic, we implemented a work-around in RewireWeightGradInputs(), which clones the original weight grad computation, delays the clone to the next iteration, and copies the input (captured in the original weight grad computation in the previous iteration) to the clone running concurrently with the forward computations. Because the clone is actually scheduled with the forward computation (of the next iteration), XLA correctly allocates its memory block which should not overlap with the memory area used by the forward computation. Then because we copy the actual input captured in the previous iteration, the clone computes the correct weight gradients. Note that most of the scheduling enforcement (that does not require scheduling kernels across iterations) is done in **B** above.



### D. Executing the captured CUDA Graph
* Source code: tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc (line: 631–719)

* GIthub link: [gpu_device.cc](https://github.com/mlsys-seo/ooo-backprop/blob/9297229f2d8ef1ebf28a507ab38d16e28639f32d/tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc#L631)

We modified the function BaseGPUDevice::Compute() to execute the CUDA Graph if the kernel is our custom kernel representing the captured CUDA Graph.


## Optimized Scheduling Algorithms for Distributed Training

Reverse first-k scheduling for data-parallel training is implemented in [dp_schedule.py](https://github.com/mlsys-seo/ooo-backprop/blob/1838adf5e780105ff17223b83d95bdc8af34adb2/expr/data_par/code/OOO_backprop/dp_schedule.py#L179), which is used by all the data-parallel training code.
The scheduling for pipeline-parallel training is in [ModelScheduleHelper.py::schedule()](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/schedule/ModelScheduleHelper.py#L32). Particularly, gradient fast-forwarding is implemented in function [\_schedule_ooo_backpropagation](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/schedule/ModelScheduleHelper.py#L277). 
Modulo allocation is applied in BERT/GPT model file; specifically we compute the device id [here](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/modeling.py#L867) and allocate the layers of the model to the device [here](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/modeling.py#L874).
