/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_

#include <memory>
#include <string>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/service/buffer_assignment.h"
#include "tensorflow/compiler/xla/service/executable.h"
#include "tensorflow/compiler/xla/service/gpu/buffer_allocations.h"
#include "tensorflow/compiler/xla/service/gpu/gpu_types.h"
#include "tensorflow/compiler/xla/service/gpu/stream_assignment.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/gpu/thunk_schedule.h"
#include "tensorflow/compiler/xla/service/hlo_dataflow_analysis.h"
#include "tensorflow/compiler/xla/service/hlo_execution_profile.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/shaped_buffer.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"
#include "tensorflow/stream_executor/device_memory_allocator.h"

namespace xla {
namespace gpu {

// GPU-targeting implementation of the XLA Executable interface.
//
// Launches the given GPU kernel via the StreamExecutor.
//
// This is an immutable data type after initialization, and thus thread safe.
class GpuExecutable : public Executable {
 public:
  struct ConstantInfo {
    std::string symbol_name;
    xla::Literal content;
    int allocation_index = -1;
  };

  // We need to share ownership of hlo_module and assignment with profiler to
  // safely keep a reference to these objects during tracing period, thus they
  // are passed as shared pointers.
  GpuExecutable(const string& text, const std::vector<uint8>& binary,
                GpuVersion gpu_version,
                std::unique_ptr<const ThunkSchedule> thunk_schedule,
                std::shared_ptr<HloModule> hlo_module,
                std::shared_ptr<const BufferAssignment> assignment,
                std::unique_ptr<HloProfilePrinterData> hlo_profile_printer_data,
                std::unique_ptr<HloProfileIndexMap> hlo_profile_index_map,
                std::vector<ConstantInfo> constants);
  ~GpuExecutable() override;

  int64 SizeOfGeneratedCodeInBytes() const override;

  // This should be called after set_ir_module_string.
  const string& ir_module_string() const { return ir_module_string_; }

  // This should be called before ExecuteOnStream.
  void set_ir_module_string(const string& ir_module_string) {
    ir_module_string_ = ir_module_string;
  }

  // Returns the compiled code for the computation. The compiled code is PTX in
  // Cuda and unused empty string in ROCm.
  const string& text() const { return text_; }

  // Returns the binary stored in this GpuExecutable. The binary is cubin in
  // Cuda, and HSA code object in ROCm. It may be empty, in which case
  // compilation is left up to the GPU driver.
  const std::vector<uint8>& binary() const { return binary_; }

  // ExecuteAsyncOnStream will fail if the compute capability of the stream
  // doesn't match the compute capability passed to this object's constructor.
  StatusOr<ExecutionOutput> ExecuteAsyncOnStream(
      const ServiceExecutableRunOptions* run_options,
      std::vector<ExecutionInput> arguments,
      HloExecutionProfile* hlo_execution_profile) override;

  std::shared_ptr<const BufferAssignment> GetBufferAssignment() const {
    return assignment_;
  }

 private:
  // If `block_host_until_done` is false, execution will not block the host
  // until the kernels have completed. This is used as an optimization for
  // clients, such as Tensorflow, that use a single stream of execution for
  // computations, and allow host-side deallocation from the allocator before
  // GPU execution completes.
  Status ExecuteThunks(const ServiceExecutableRunOptions* run_options,
                       const BufferAllocations& buffer_allocations,
                       bool block_host_until_done,
                       HloExecutionProfile* hlo_execution_profile);

  Status ExecuteThunksAndGraphCapture(const ServiceExecutableRunOptions* run_options,
                                      const BufferAllocations& buffer_allocations,
                                      bool block_host_until_done,
                                      HloExecutionProfile* hlo_execution_profile);

  void RewireWeightGradInputs( const BufferAllocations& buffer_allocations );

  // Returns the value set of the root instruction of the entry
  // computation. Uses dataflow analysis from buffer assignment.
  const InstructionValueSet& GetRootValueSet() const;

  using BufferAllocToDeviceMemoryMap =
      absl::flat_hash_map<BufferAllocation::Index, se::DeviceMemoryBase>;

  // Loads the PTX or CUBIN for this executable into `executor` and resolves the
  // globals corresponding to constant buffers.  Returns a map mapping buffer
  // allocation indices to GPU pointers.
  StatusOr<const BufferAllocToDeviceMemoryMap*> ResolveConstantGlobals(
      stream_executor::Stream* stream);

  // GpuExecutable check with either AMD's ISA version, or Nvidia's major minor
  // version for compute capability, depending on the hardware.
  Status CheckCompatibilityWithServiceExecutableRunOptions(
      const ServiceExecutableRunOptions* run_options);

  StatusOr<BufferAllocations> GenerateBufferAllocations(
      absl::Span<ExecutionInput const> arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      se::DeviceMemoryAllocator* const memory_allocator,
      se::StreamExecutor* executor);

  StatusOr<se::DeviceMemoryBase> BufferForAllocation(
      absl::Span<ExecutionInput const> arguments,
      const GpuExecutable::BufferAllocToDeviceMemoryMap* globals,
      const BufferAllocation& allocation,
      se::DeviceMemoryAllocator* const memory_allocator, int device_ordinal,
      int64 arg_idx);

  // The LLVM IR, in string format, of the unoptimized module generated for
  // this GpuExecutable. We save a string instead of an llvm::Module* because
  // leaving llvm::Module* in a singleton can cause the heap checker to emit
  // false positives.
  //
  // This string should be modified only before ExecuteOnStream.
  string ir_module_string_;

  // The compiled code for the computation.
  const string text_;

  // The GPU machine code for the computation, targeting GPUs at
  // compute_capability_.
  //
  // May be empty, in which case we leave compilation up to the GPU driver.
  const std::vector<uint8> binary_;

  // The GPU version for compute compatibility check.
  GpuVersion gpu_version_;

  // The thunks to be invoked by this GpuExecutable. They are generated by the
  // IrEmitter.
  const std::unique_ptr<const ThunkSchedule> thunk_schedule_;

  // Owns the buffer data at runtime. It provides information to allocate
  // memory for every output/temp buffers.
  const std::shared_ptr<const BufferAssignment> assignment_;

  // Cache of module handles and constant buffer allocation maps used by
  // `ResolveConstantGlobals`.
  tensorflow::mutex module_handle_mutex_;
  std::map<stream_executor::StreamExecutor*, se::ScopedModuleHandle>
      module_handles_ TF_GUARDED_BY(module_handle_mutex_);
  std::map<stream_executor::StreamExecutor*, BufferAllocToDeviceMemoryMap>
      module_globals_ TF_GUARDED_BY(module_handle_mutex_);

  std::vector<ConstantInfo> constants_;

  bool do_ooo_backprop_ = false;
  bool is_main_executable_ = false;
  int execution_count_ = 0;
  int capture_iter_ = -1;

  std::vector<std::string> origin_w_grad_names;
  std::vector<void*> origin_w_grad_input1;
  std::vector<void*> origin_w_grad_input2;
  std::vector<void*> origin_wgrad_output;
  
  std::vector<std::string> new_w_grad_names;
  std::vector<void*> new_w_grad_input1;
  std::vector<void*> new_w_grad_input2;
  std::vector<void*> new_wgrad_output;
  
  std::vector<size_t> w_grad_input1_sizes;
  std::vector<size_t> w_grad_input2_sizes;
  std::vector<size_t> w_grad_output_size;
  
  std::string FORWARD_GRAPH = "FIRST_Graph";
  std::string FORWARD_OVERLAP_GRAPH = "FORWARD_OVERLAP_WGRADS";
  std::string DEFAULT_GRAPH = "LAST_GRAPH";

  std::string overlap_start_name_;
  std::string overlap_end_name_;

  bool is_overlap_w_grad_op( std::string op_name, std::string hlo_name  ){
    if (overlap_start_name_ == "NONE") {
      return false;
    }

    if ( op_name.find(overlap_start_name_) != std::string::npos && 
        op_name.find("Conv2DBackpropFilter") != std::string::npos &&
        op_name.find("Dummy") == std::string::npos &&
        hlo_name.find("custom") != std::string::npos ) {
      return true;
    } else {
      return false;
    }
  }

  bool is_remain_graph_start( std::string op_name  ){
    if (overlap_end_name_ == "NONE") {
      return false;
    }

    if ( op_name.find(overlap_end_name_) != std::string::npos ) {
      return true;
    } else {
      return false;
    }
  }

  bool is_deleted_op( std::string op_name ){
    if (op_name.find("DummyConv2DBackpropFilter") != std::string::npos) {
        return true;
    } else {
        return false;
    }
  }



  TF_DISALLOW_COPY_AND_ASSIGN(GpuExecutable);
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_GPU_EXECUTABLE_H_
