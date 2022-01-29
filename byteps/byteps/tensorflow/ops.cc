// Copyright 2016 The TensorFlow Authors. All Rights Reserved.
// Modifications copyright (C) 2018 Uber Technologies, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#include <memory>
#include <queue>
#include <thread>
#include <unordered_map>

#include "ops.h"

using namespace byteps;

namespace byteps {
namespace tensorflow {



// modified for OOO-Backprop

AsyncCommManager communication_manager;
AsyncCommLockManager lock_manager;

// const char *tmp = getenv("DEBUG_C_PRINT");
// string env_var(tmp ? tmp : "");
// bool DEBUG_PRINT = !env_var.empty();
bool DEBUG_PRINT = false;

// end modified for OOO-Backprop
namespace {

::tensorflow::Status ConvertStatus(const common::Status& status) {
  switch (status.type()) {
    case common::OK:
      return ::tensorflow::Status::OK();
    case common::UNKNOWN_ERROR:
      return ::tensorflow::errors::Unknown(status.reason());
    case common::PRECONDITION_ERROR:
      return ::tensorflow::errors::FailedPrecondition(status.reason());
    case common::ABORTED:
      return ::tensorflow::errors::Aborted(status.reason());
    case common::INVALID_ARGUMENT:
      return ::tensorflow::errors::InvalidArgument(status.reason());
    default:
      return ::tensorflow::errors::Unknown("Unknown error.");
  }
}

int GetDeviceID(::tensorflow::OpKernelContext* context) {
  int device = CPU_DEVICE_ID;
  if (context->device() != nullptr &&
      context->device()->tensorflow_gpu_device_info() != nullptr) {
    device = context->device()->tensorflow_gpu_device_info()->gpu_id;
  }
  return device;
}

// Define all types for TensorUtil.
const common::DataType ConvertDType(int dtype) {
  switch (dtype) {
    case ::tensorflow::DT_UINT8:
      return common::BYTEPS_UINT8;
    case ::tensorflow::DT_INT8:
      return common::BYTEPS_INT8;
    // case ::tensorflow::DT_UINT16:
    //   return common::BYTEPS_UINT16;
    // case ::tensorflow::DT_INT16:
    //   return common::BYTEPS_INT16;
    case ::tensorflow::DT_INT32:
      return common::BYTEPS_INT32;
    case ::tensorflow::DT_INT64:
      return common::BYTEPS_INT64;
    case ::tensorflow::DT_HALF:
      return common::BYTEPS_FLOAT16;
    case ::tensorflow::DT_FLOAT:
      return common::BYTEPS_FLOAT32;
    case ::tensorflow::DT_DOUBLE:
      return common::BYTEPS_FLOAT64;
    // case ::tensorflow::DT_BOOL:
    //   return common::BYTEPS_BOOL;
    default:
      throw std::logic_error("Invalid tensor type.");
  }
}

}  // namespace

TFReadyEvent::TFReadyEvent(::tensorflow::DeviceContext* device_context) {
  auto executor = device_context->stream()->parent();
  auto ready_event = new perftools::gputools::Event(executor);
  ready_event->Init();
  device_context->stream()->ThenRecordEvent(ready_event);
  event_ = std::shared_ptr<perftools::gputools::Event>(ready_event);
}

bool TFReadyEvent::Ready() const {
  return event_->PollForStatus() !=
         perftools::gputools::Event::Status::kPending;
}

TFTensor::TFTensor(::tensorflow::Tensor& tensor) : tensor_(tensor) {}

const common::DataType TFTensor::dtype() const {
  return ConvertDType(tensor_.dtype());
}

const common::TensorShape TFTensor::shape() const {
  common::TensorShape shape;
  for (auto dim : tensor_.shape()) {
    shape.AddDim(dim.size);
  }
  return shape;
}

const void* TFTensor::data() const {
  return (const void*)tensor_.tensor_data().data();
}

int64_t TFTensor::size() const { return (int64_t)tensor_.tensor_data().size(); }

// On GPU this event will signal that data is ready, and tensors are
// allocated.
common::ReadyEvent* RecordReadyEvent(::tensorflow::OpKernelContext* context) {
  auto device_context = context->op_device_context();
  if (device_context != nullptr) {
    return new TFReadyEvent(device_context);
  }
  return nullptr;
}

extern "C" void byteps_tensorflow_declare_tensor(char* name) {
  std::string tensor_name(name);
  common::IsTensorDeclared(tensor_name);
  return;
}

void StartTask(::tensorflow::OpKernelContext* context,
               ::tensorflow::AsyncOpKernel::DoneCallback done,
               std::string node_name, std::shared_ptr<TFTensor> byteps_input,
               std::shared_ptr<TFTensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event) {
  auto& byteps_context = common::GetContextFromName(node_name);
  auto device = GetDeviceID(context);
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);
  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  // TODO: assign priority based on topological sort
  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [context, done](const common::Status& status) {
                      context->SetStatus(ConvertStatus(status));
                      done();
                    },
                    queue_list);
  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
}

class BytePSPushPullOp : public ::tensorflow::AsyncOpKernel {
  private:
     std::string input_tensor_name;
  public:
    explicit BytePSPushPullOp(::tensorflow::OpKernelConstruction* context)
        : AsyncOpKernel(context) {
            context->GetAttr("input_name", &input_tensor_name);
        }

    void ComputeAsync(::tensorflow::OpKernelContext* context,
                      DoneCallback done) override {
      OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                          done);

      auto tensor = context->input(0);
      ::tensorflow::Tensor* output;
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, tensor.shape(), &output), done);
      // ReadyEvent makes sure input tensor is ready, and output is allocated.
      auto ready_event =
          std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
      auto bps_input = std::make_shared<TFTensor>(tensor);
      auto bps_output = std::make_shared<TFTensor>(*output);
      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
          tmp_name = node_name;
      } else {
          tmp_name = input_tensor_name;
      }
      auto& bps_context = common::GetContextFromName(tmp_name);
      if (bps_context.initialized) {
        StartTask(context, done, tmp_name, bps_input, bps_output, ready_event);
      } else {
        std::thread t(StartTask, context, done, tmp_name, bps_input, bps_output,
                      ready_event);
        t.detach();
      }
    }
};

REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(::tensorflow::DEVICE_CPU),
                        BytePSPushPullOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPull").Device(::tensorflow::DEVICE_GPU),
                        BytePSPushPullOp);

REGISTER_OP("BytepsPushPull")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_name: string = 'default_tensor_name'")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
Perform an PushPull on a tensor. All other processes that do a reduction
on a tensor with the same name must have the same dimension for that tensor.
Tensors are reduced with other tensors that have the same node name for the
push_pull.
Arguments
    tensor:     A tensor to reduce.
Output
    sum:    A tensor with the same shape as `tensor`, summed across all processes.
)doc");

// modified for OOO-Backprop
::tensorflow::TensorShape get_output_shape(int s0, int s1, int s2, int s3) {
  if (s1 == 0) {
    ::tensorflow::TensorShape shape_({s0});
      return shape_;
  }
  else if(s2 == 0) {
    ::tensorflow::TensorShape shape_({s0, s1});
    return shape_;
  }
  else {
    ::tensorflow::TensorShape shape_({s0, s1, s2, s3});
    return shape_;
  }
}

void StartTaskAsync(::tensorflow::OpKernelContext* context,
               ::tensorflow::AsyncOpKernel::DoneCallback done,
               std::string node_name, std::shared_ptr<TFTensor> byteps_input,
               std::shared_ptr<TFTensor> byteps_output,
               std::shared_ptr<common::ReadyEvent> ready_event, int op_index) {
  
  auto& byteps_context = common::GetContextFromName(node_name);
  auto device = GetDeviceID(context);
  auto size = byteps_input->size();
  auto dtype = byteps_input->dtype();
  void* cpubuff = (device == CPU_DEVICE_ID)
                      ? const_cast<void*>(byteps_input->data())
                      : nullptr;
  common::InitTensor(byteps_context, size, dtype, cpubuff);

  auto queue_list = common::GetPushQueueList(device);
  auto queue_list_pull = common::GetPullQueueList(device);

  queue_list->insert(queue_list->end(), queue_list_pull->begin(),
                     queue_list_pull->end());

  if (DEBUG_PRINT) {
    std::cout << "[OOO-LOG:: C] Async Task " << node_name << " start |"
              << " index: " << op_index 
              << " priority: " << -byteps_context.declared_key
              << "\n";
  }
  
  // delete after wait
  if( communication_manager.is_valid_savedTensors(op_index) )
  {
    communication_manager.del_savedTensors(op_index);
  }

  ::tensorflow::Tensor* tensor = new ::tensorflow::Tensor( (*byteps_output).tensor_ );
  communication_manager.set_savedTensors(op_index, tensor);
  
  lock_manager.lock(op_index);

  auto enqueue_result =
      EnqueueTensor(byteps_context, byteps_input, byteps_output, ready_event,
                    device, -byteps_context.declared_key, 0,
                    [context, done, node_name, op_index](const common::Status& status) {
                        if (DEBUG_PRINT) {
                          std::cout << "[OOO-LOG:: C] Async Task " << node_name << " DONE |"
                          << " index: " << op_index
                          << "\n";
                        }
                        lock_manager.unlock(op_index);
                    },
                    queue_list);

  OP_REQUIRES_OK_ASYNC(context, ConvertStatus(enqueue_result), done);
  
  if(DEBUG_PRINT){
    std::cout << "[C-PUSH] TASK END " << node_name << "\n";
  }
  
  context->SetStatus( ConvertStatus(common::Status::OK()) );
  done();
}

class BytePSPushPullAsyncOp : public ::tensorflow::AsyncOpKernel {
  private:
    std::string input_tensor_name;
    std::string debug_tensor_name;
    int op_index = -1;

  public:
    explicit BytePSPushPullAsyncOp(::tensorflow::OpKernelConstruction* context)
        : AsyncOpKernel(context) {
            context->GetAttr("input_name", &input_tensor_name);
            context->GetAttr("op_index", &op_index);
            context->GetAttr("debug_name", &debug_tensor_name);

            if (DEBUG_PRINT) {
              std::cout << "[OOO-LOG:: C] BytePSPushPullAsyncOp INIT | "
              << " index: " << op_index 
              << " debug_tensor_name: " << debug_tensor_name
              << "\n";
            }
      }

    void ComputeAsync(::tensorflow::OpKernelContext* context,
                      DoneCallback done) override {
      OP_REQUIRES_OK_ASYNC(context, ConvertStatus(common::CheckInitialized()),
                          done);

      auto tensor = context->input(0);
      ::tensorflow::Tensor* output;
      
      OP_REQUIRES_OK_ASYNC(
          context, context->allocate_output(0, tensor.shape(), &output), done);
      
      // ReadyEvent makes sure input tensor is ready, and output is allocated.
      auto ready_event =
          std::shared_ptr<common::ReadyEvent>(RecordReadyEvent(context));
      auto bps_input = std::make_shared<TFTensor>(tensor);
      auto bps_output = std::make_shared<TFTensor>(*output);

      //auto bps_output = std::make_shared<TFTensor>(tmp_buff);
      auto node_name = name();
      std::string tmp_name;
      if (input_tensor_name == "default_tensor_name") {
          tmp_name = node_name;
      } else {
          tmp_name = input_tensor_name;
      }

      auto& bps_context = common::GetContextFromName(tmp_name);

      if (bps_context.initialized) {
        StartTaskAsync(context, done, tmp_name, bps_input, bps_output, ready_event, op_index);
      } else {
        std::thread t(StartTaskAsync, context, done, tmp_name, bps_input, bps_output, ready_event, op_index);
        t.detach();
      }
    }
};

REGISTER_KERNEL_BUILDER(Name("BytepsPushPullAsync").Device(::tensorflow::DEVICE_CPU),
                        BytePSPushPullAsyncOp);
REGISTER_KERNEL_BUILDER(Name("BytepsPushPullAsync").Device(::tensorflow::DEVICE_GPU),
                        BytePSPushPullAsyncOp);

REGISTER_OP("BytepsPushPullAsync")
    .Attr("T: {int32, int64, float16, float32, float64}")
    .Attr("input_name: string = 'default_tensor_name'")
    .Attr("debug_name: string = 'debug_name'")
    .Attr("op_index: int")
    .Input("tensor: T")
    .Output("sum: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return ::tensorflow::Status::OK();
    })
    .Doc(R"doc(
      Perform an PushPull on a tensor. All other processes that do a reduction
      on a tensor with the same name must have the same dimension for that tensor.
      Tensors are reduced with other tensors that have the same node name for the
      push_pull.
      Arguments
          tensor:     A tensor to reduce.
      Output
          sum:    A tensor with the same shape as `tensor`, summed across all processes.
      )doc");

class PollingDataOp : public ::tensorflow::OpKernel {
public:
  int op_index = -1;
  int s0, s1, s2, s3 = 0;

  explicit PollingDataOp( ::tensorflow::OpKernelConstruction* context) : OpKernel(context) {
    context->GetAttr("op_index", &op_index);
    context->GetAttr("s0", &s0);
    context->GetAttr("s1", &s1);
    context->GetAttr("s2", &s2);
    context->GetAttr("s3", &s3);

    if (DEBUG_PRINT) {
      std::cout << "[OOO-LOG:: C-PollingDataOp] INIT |"
                << " index: " << op_index 
                << " shape: {" << s0 << "," << s1 << "," << s2 << "," << s3 << "}"
                << "\n";
    }

    lock_manager.unlock(op_index); //unlock
  }

  void Compute( ::tensorflow::OpKernelContext* context ) override {
    // if (DEBUG_PRINT) {
    //   std::cout << "[OOO-LOG:: C-PollingDataOp] START |"
    //             << " index: " << op_index
    //             << "\n";
    // }

    while (lock_manager.is_locked(op_index)) {
        usleep(1);
    }

    // if (DEBUG_PRINT) {
    //   std::cout << "[OOO-LOG:: C-PollingDataOp] END |"
    //             << " index: " << op_index
    //             << "\n";
    // }

    auto shape_ = get_output_shape(s0, s1, s2, s3);
    ::tensorflow::Tensor* output = nullptr;
    context->set_output(0, *( communication_manager.get_savedTensors(op_index) ));

    OP_REQUIRES_OK( context, context->allocate_output(0, shape_, &output));
  }
};

REGISTER_KERNEL_BUILDER(Name("PollingData").Device( ::tensorflow::DEVICE_GPU), PollingDataOp);

REGISTER_OP("PollingData")
  .Attr("T: {int32, int64, float16, float32, float64}")
  .Attr("op_index: int")
  .Attr("s0: int")
  .Attr("s1: int")
  .Attr("s2: int")
  .Attr("s3: int")
  .Input("index: T")
  .Output("output: float32")
  .Doc(R"doc( Communication kernal for polling data )doc");


class CommTerminalOp : public ::tensorflow::OpKernel {
  public:
      explicit CommTerminalOp( ::tensorflow::OpKernelConstruction* context) : OpKernel(context) {}
      void Compute(::tensorflow::OpKernelContext* context ) override {}
};

REGISTER_KERNEL_BUILDER(Name("CommTerminal").Device(::tensorflow::DEVICE_GPU), CommTerminalOp);

REGISTER_OP("CommTerminal")
  .Input("input_tensor: float32")
  .Doc(R"doc( eatdata... )doc");

// end modified for OOO-Backprop



}  // namespace tensorflow
}  // namespace byteps
