# Overview

Our optimized scheduling algorithms for data-parallel and pipeline-parallel training are implemented entirely in Python, which is described at the botton of this page.
For the single-GPU training, our optimization, that is, multi-stream ooo computation, requires modifications in TensorFlow, which we first describe below.

## Changes in TensorFlow for single-GPU Training

The source code modification in TensorFlow XLA is for multi-stream out-of-order computation (Section 4.1) and pre-compiled kernel issue (Section 4.2) for single-GPU training. The following (A–D) are the locations and descriptions of our modifications.

### A. Stream allocation

 * Source code: tensorflow/tensorflow/compiler/xla/service/gpu/stream_assignment.cc (line: 81–100)

* Github link: [stream_assignment.cc](https://github.com/mlsys-seo/ooo-backprop/blob/4a6932a87837a1a825f511889800e1f941be08d2/tensorflow/tensorflow/compiler/xla/service/gpu/stream_assignment.cc#L81)

We modified the function AssignStreamToHlo() to assign the streams to kernels as they are annotated in our Python-level scheduler, the results of which are in {densenet,mobilenet_v3}_schedule_map.py.



### B. Enforcing the kernel schedule for main- and sub-streams
* Source code: tensorflow/tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc (line: 435–440)

* Github link: [gpu_hlo_schedule.cc](https://github.com/mlsys-seo/ooo-backprop/blob/4a6932a87837a1a825f511889800e1f941be08d2/tensorflow/tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.cc#L435)

We modified the function GpuHloSchedule::Build() to invoke our implemented function MakeOOOLaunchOrder(), which is in line 298. The function enforces the execution schedule for the kernels in main-stream and sub-stream. Note that the kernel scheduling is already determined and stored in {densenet,mobilenet_v3}_schedule_map.py and here we simply enforce the determined schedule.




### C. CUDA Graph Capturing
* Source code: tensorflow/tensorflow/compiler/xla/service/gpu/gpu_executable.cc (line: 817–836)

* Github link: [gpu_executable.cc](https://github.com/mlsys-seo/ooo-backprop/blob/17de9d83176d54abebd8da597ef169524dfa281b/tensorflow/tensorflow/compiler/xla/service/gpu/gpu_executable.cc#L817)


We modified the function GpuExecutable::ExecuteAsyncOnStream() to invoke our implemented functions, i.e., RewireWeightGradInputs() and ExecuteThunksAndGraphCapture(). The function ExecuteThunksAndGraphCapture() (in line 158) is modified from the function ExecuteThunks() (in line 317) and it has the code for CUDA graph capturing (line 214, 274–280). 

The function RewireWeightGradInputs() (in line 629) is applied if a weight gradient computation is scheduled to overlap with a forward computation in the next iteration. Because XLA performs the scheduling within a single iteration, if we delay a weight grad computation to the next iteration, its memory allocation logic (for multi-stream) does not work. To re-use the existing memory allocation logic, we implemented a work-around in RewireWeightGradInputs(), which copies a dummy weight grad computation, delays it to the next iteration, and rewires the input (in the previous iteration) to the overlapping weight grad computation. Note that most of the scheduling enforcement (that does not require scheduling kernels across iterations) is done in **B** above.



### D. Executing the captured CUDA Graph
* Source code: tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc (line: 611–633)

* GIthub link: [gpu_device.cc](https://github.com/mlsys-seo/ooo-backprop/blob/17de9d83176d54abebd8da597ef169524dfa281b/tensorflow/tensorflow/core/common_runtime/gpu/gpu_device.cc#L611)

We modified the function BaseGPUDevice::Compute() to execute the CUDA Graph if the kernel is our custom kernel representing the captured CUDA Graph.


## Optimized Scheduling Algorithms for Distributed Training

Reverse first-k scheduling for data-parallel training is implemented in [dp_schedule.py](https://github.com/mlsys-seo/ooo-backprop/blob/1838adf5e780105ff17223b83d95bdc8af34adb2/expr/data_par/code/OOO_backprop/dp_schedule.py#L179), which is used by all the data-parallel training code.
The scheduling for pipeline-parallel training is in [ModelScheduleHelper.py::schedule()](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/schedule/ModelScheduleHelper.py#L32). Particularly, gradient fast-forwarding is implemented in function [\_schedule_ooo_backpropagation](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/schedule/ModelScheduleHelper.py#L277). 
Modulo allocation is applied in BERT/GPT model file; specifically we compute the device id [here](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/modeling.py#L867) and allocate the layers of the model to the device [here](https://github.com/mlsys-seo/ooo-backprop/blob/396caa1a68738884af6a2199ca91bf4681043c93/expr/pipe_par/code/OOO_backprop/modeling.py#L874).
