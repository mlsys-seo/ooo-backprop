// Copyright 2019 Bytedance Inc. or its affiliates. All Rights Reserved.
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

#ifndef BYTEPS_TENSORFLOW_OPS_H
#define BYTEPS_TENSORFLOW_OPS_H

#include <string>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

#define EIGEN_USE_THREADS
#include "tensorflow/stream_executor/stream.h"

#include "../common/operations.h"

// modified for OOO-Backprop
#define LOCK_FLAG 0
#define LOCKS_SIZE 1024
#define BUFF_SIZE 60
#define SAVED_TENSOR_SIZE 500
// end modified for OOO-Backprop

namespace byteps {
namespace tensorflow {

class TFReadyEvent : public common::ReadyEvent {
 public:
  TFReadyEvent(::tensorflow::DeviceContext* device_context);
  bool Ready() const override;

 private:
  std::shared_ptr<perftools::gputools::Event> event_;
};

class TFTensor : public common::Tensor {
 public:
  TFTensor(::tensorflow::Tensor& tensor);
  virtual const common::DataType dtype() const override;
  virtual const common::TensorShape shape() const override;
  virtual const void* data() const override;
  virtual int64_t size() const override;
  ::tensorflow::Tensor tensor_;

//  protected:
//   ::tensorflow::Tensor tensor_;
};

extern "C" void byteps_tensorflow_declare_tensor(char* name);

// modified for OOO-Backprop
struct OOOLock {
    int lock = LOCK_FLAG;
    char buffer[BUFF_SIZE];
};

class AsyncCommLockManager {
 private:
  volatile struct OOOLock locks[LOCKS_SIZE];

 public:
  int is_locked(int idx) {
      return locks[idx].lock;
  }

  void lock(int idx)  {
      locks[idx].lock = 1;
  }

  void unlock(int idx) {
      locks[idx].lock = 0;
  }
};

class AsyncCommManager {
 private:
  ::tensorflow::Tensor* savedTensors[SAVED_TENSOR_SIZE] = {0};

 public:
  ::tensorflow::Tensor* get_savedTensors(int idx) {
      return savedTensors[idx];
  }

  void set_savedTensors(int idx, ::tensorflow::Tensor* value) {
      savedTensors[idx] = value;
  }

  bool is_valid_savedTensors(int idx) {
      return bool(savedTensors[idx]);
  }

  void del_savedTensors(int idx) {
      delete savedTensors[idx];
      savedTensors[idx] = 0;
  }

};

// end modified for OOO-Backprop

}  // namespace tensorflow
}  // namespace byteps

#endif  // BYTEPS_TENSORFLOW_OPS_H
