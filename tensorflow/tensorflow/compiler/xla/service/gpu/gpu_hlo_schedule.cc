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

#include <deque>
#include <memory>
#include <unordered_map>

#include "tensorflow/compiler/xla/service/gpu/gpu_hlo_schedule.h"

#include "absl/memory/memory.h"
#include "tensorflow/compiler/xla/service/buffer_value.h"
#include "tensorflow/compiler/xla/service/hlo_memory_scheduler.h"
#include "tensorflow/compiler/xla/service/hlo_reachability.h"
#include "tensorflow/compiler/xla/service/hlo_schedule.h"
#include "tensorflow/compiler/xla/types.h"

namespace xla {
namespace gpu {

namespace {

// An HLO partial ordering based on the actual stream assignment and thunk
// launch order.
class GpuHloOrdering : public PredecessorHloOrdering {
 public:
  GpuHloOrdering(const HloModule* module,
                 const StreamAssignment& stream_assignment,
                 const std::vector<HloInstruction*>& thunk_launch_order);
  ~GpuHloOrdering() override = default;

  // Only the entry computation can possibly be sequentially ordered, and only
  // if we've assigned all instructions to a single stream.
  const HloInstructionSequence* SequentialOrder(
      const HloComputation& computation) const override {
    return &computation == module_->entry_computation() ? entry_sequence_.get()
                                                        : nullptr;
  }

  string ToString() const override { return ToStringHelper("GpuHloOrdering"); }

 private:
  std::unique_ptr<HloInstructionSequence> entry_sequence_;
};

GpuHloOrdering::GpuHloOrdering(
    const HloModule* module, const StreamAssignment& stream_assignment,
    const std::vector<HloInstruction*>& thunk_launch_order)
    : PredecessorHloOrdering(module) {
  // The entry computation has a total order when there's only one stream.
  if (stream_assignment.StreamCount() == 1) {
    entry_sequence_ =
        absl::make_unique<HloInstructionSequence>(thunk_launch_order);
  }

  // The ordering of instructions for the entry computation is determined by the
  // total order of thunk launches, and stream assignment. Instructions are
  // sequential within a stream and concurrent across streams. In addition, the
  // GpuExecutable adds cross-stream dependency edges to ensure each instruction
  // waits for its operands before executing.
  //
  // The predecessor map is built incrementally, in thunk launch order. We
  // record the most-recently seen instructions per stream in
  // 'last_instruction_per_stream'. This lets us quickly determine the
  // same-stream predecessors of each instruction.

  // Compute the set of all instructions we will want to set reachability on.
  auto predecessor_map = absl::make_unique<HloReachabilityMap>(
      module->entry_computation()->MakeInstructionPostOrder());

  // The most recently visited instruction per stream.
  std::vector<const HloInstruction*> last_instruction_per_stream(
      stream_assignment.StreamCount(), nullptr);

  for (const HloInstruction* hlo : thunk_launch_order) {
    predecessor_map->SetReachable(hlo, hlo);
    if (stream_assignment.HasStreamAssigned(*hlo)) {
      // Gather all instruction which are immediate predecessors of 'hlo' in the
      // reachability graph.
      std::vector<const HloInstruction*> immediate_preds;
      immediate_preds.insert(immediate_preds.end(), hlo->operands().begin(),
                             hlo->operands().end());
      immediate_preds.insert(immediate_preds.end(),
                             hlo->control_predecessors().begin(),
                             hlo->control_predecessors().end());

      // All ops already queued on the same instruction stream, and their
      // transitive predecessors, are predecessors.
      const int stream_no = stream_assignment.StreamNumberForHlo(*hlo);
      if (last_instruction_per_stream[stream_no] != nullptr) {
        immediate_preds.push_back(last_instruction_per_stream[stream_no]);
      }
      predecessor_map->FastSetReachabilityToUnion(immediate_preds, hlo);
      last_instruction_per_stream[stream_no] = hlo;
    } else {
      // Only parameters and constants don't have an assigned stream, since they
      // don't require a thunk. These ops don't have any predecessors.
      CHECK(hlo->opcode() == HloOpcode::kParameter ||
            hlo->opcode() == HloOpcode::kConstant);
      CHECK_EQ(hlo->operand_count(), 0);
    }
  }
  predecessors_.emplace(module->entry_computation(),
                        std::move(predecessor_map));

  // The ordering of instructions in subcomputations is based solely on control
  // and data dependencies.
  //
  // TODO(toddw): Each subcomputation is actually emitted as a function in DFS
  // postorder, so we can do better and establish the total order here. We don't
  // do that yet since it's hard to ensure that the order here is the order used
  // by IrEmitterNested. And mismatched ordering bugs would be hard to find.
  for (auto* computation : module->computations()) {
    if (computation != module->entry_computation() &&
        !computation->IsFusionComputation()) {
      predecessors_.emplace(computation,
                            HloReachabilityMap::Build(computation));
    }
  }
}

// Computes a topological launch_order that is close to a breadth-first
// order. This heuristic works well for graphs where concurrent kernels are
// located at the same layer. It can often reduce dependency between concurrent
// GEMMs due to intra-stream total orders.  E.g. consider the following HLO
// graph where the numbers in the parens indicate the stream assigned to each
// HLO.
//
//   A(0) -> D(0) -> E(1)
//    |
//    v
//   B(0)
//    |
//    v
//   C(0)
//
// If the total order is A,B,C,D,E, then C and E would be sequentialized
// because C completes before D starts in stream 0, and E depends on D.
// However, if the total order is A,B,D,C,E, then C and E can run
// concurrently.
void BFSLaunchOrder(const HloComputation* computation,
                    std::vector<HloInstruction*>* launch_order) {
  // This topological sort uses two data structures:
  // 1. `incoming_edge_count` which keeps track of the number of incoming
  // edges to each HLO;
  // 2. `queue` which contains all HLOs with no incoming edges.
  //
  // The sorting algorithm repeatedly pops the top from the queue and deletes
  // that HLO from the graph, making more HLOs incoming-edge free.
  std::deque<HloInstruction*> queue;
  std::unordered_map<const HloInstruction*, int64> incoming_edge_count;
  for (auto* hlo : computation->instructions()) {
    if (hlo->operand_count() == 0) {
      queue.push_back(hlo);
    } else {
      incoming_edge_count[hlo] =
          std::set<HloInstruction*>(hlo->operands().begin(),
                                    hlo->operands().end())
              .size();
    }
  }

  while (!queue.empty()) {
    HloInstruction* x = queue.front();
    queue.pop_front();
    launch_order->push_back(x);
    for (HloInstruction* y : x->users()) {
      --incoming_edge_count[y];
      if (incoming_edge_count[y] == 0) {
        queue.push_back(y);
      }
    }
  }
  
  std::cout << "####### launch order ############## \n";
  for (auto* hlo : *launch_order) {
      std::cout << "\t op : " << hlo->metadata().op_name() << " " << hlo->name()  << "\n";
  }
}

std::vector<std::string> StringSplit(std::string str, char delimiter) {
  std::vector<std::string> ret;
  std::stringstream ss(str);

  std::string temp;
  while (getline(ss, temp, delimiter)) {
    ret.emplace_back(temp);
  }

  return ret;
}

bool IsForwardOp(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }

  std::string op_name = hlo.metadata().op_name();
  return (op_name.find("Backprop") == std::string::npos &&
          op_name.find("Conv2D") != std::string::npos);
}

bool IsWeightGradOp(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }

  std::string op_name = hlo.metadata().op_name();
  return op_name.find("BackpropFilter") != std::string::npos;
}

bool IsOutputGradOp(const HloInstruction& hlo) {
  if (hlo.opcode() != HloOpcode::kCustomCall) {
    return false;
  }

  std::string op_name = hlo.metadata().op_name();
  return op_name.find("BackpropInput") != std::string::npos;
}

bool OverlapForward(const HloInstruction& hlo) {
  if (!IsWeightGradOp(hlo)) {
    return false;  
  }

  std::string op_name = hlo.metadata().op_name();

  return (op_name.find("FWD") != std::string::npos &&
          op_name.find("Dummy") == std::string::npos);
}

bool IsDummyForOverlapForward(const HloInstruction& hlo) {
  if (!IsWeightGradOp(hlo)) {
    return false;  
  }

  std::string op_name = hlo.metadata().op_name();

  return (op_name.find("FWD") != std::string::npos &&
          op_name.find("Dummy") != std::string::npos);
}

bool OverlapOutputGrad(const HloInstruction& hlo) {
  if (!IsWeightGradOp(hlo)) {
    return false;  
  }

  std::string op_name = hlo.metadata().op_name();

  return op_name.find("BWD") != std::string::npos;
}

bool IsNormalOp(const HloInstruction& hlo) {
  return !OverlapForward(hlo) && !IsDummyForOverlapForward(hlo) && !OverlapOutputGrad(hlo);
  //return !OverlapForward(hlo) && !OverlapOutputGrad(hlo);
}

int GetLayerId(const HloInstruction& hlo) {
  int layer_id = -1;
  std::string op_name = hlo.metadata().op_name();
  std::vector<std::string> op_name_tokens = StringSplit(op_name, '/');
  for (auto token : op_name_tokens) {
    if (token.find("CONV") != std::string::npos) {
      std::vector<std::string> token_splits = StringSplit(token, '_');
      layer_id = std::stoi(token_splits[1]);
      break;
    }
  }

  return layer_id;
}

int GetOverlapTargetLayerId(const HloInstruction& hlo) {
  if (!OverlapForward(hlo) && !OverlapOutputGrad(hlo)) {
    return -1;  
  }

  int target_layer_id = -1;
  std::string op_name = hlo.metadata().op_name();
  std::vector<std::string> op_name_tokens = StringSplit(op_name, '/');
  for (auto token : op_name_tokens) {
    std::vector<std::string> token_splits = StringSplit(token, '_');
    for (auto split : token_splits) {
      if (split.find("FWD") != std::string::npos || split.find("BWD") != std::string::npos) {
        //target_layer_id = split.back() - '0';
        int str_idx = split.find("D");
        int split_size = split.size();
        std::string split_step_2 = split.substr(str_idx+1,split_size);
        //target_layer_id = split.back() - '0';
        target_layer_id = std::stoi(split_step_2);
        break;

      }
    }
  }

  return target_layer_id;
}



void MakeOOOLaunchOrder(const HloComputation* computation,
                        std::vector<HloInstruction*>* launch_order) {
  // This topological sort uses two data structures:
  // 1. `incoming_edge_count` which keeps track of the number of incoming
  // edges to each HLO;
  // 2. `queue` which contains all HLOs with no incoming edges.
  //
  // The sorting algorithm repeatedly pops the top from the queue and deletes
  // that HLO from the graph, making more HLOs incoming-edge free.
  std::deque<HloInstruction*> queue;
  std::unordered_map<const HloInstruction*, int64> incoming_edge_count;
  auto PropagateNextNodes = [&incoming_edge_count, &queue](HloInstruction* x) {
    for (HloInstruction* y : x->users()) {
      --incoming_edge_count[y];
      if (incoming_edge_count[y] == 0) {
        queue.push_back(y);
      }
    }  
  };

  for (auto* hlo : computation->instructions()) {
    //std::cout << "op : " << hlo->metadata().op_name() << " " << hlo->name()  << "\n";
    /*
    for (HloInstruction* y : hlo->users()) {
      std::cout << "\t" << y->metadata().op_name() << " " << y->name()  << " cnt : " << incoming_edge_count[y] << "\n";
    }
    */

    if (hlo->operand_count() == 0) {
      queue.push_back(hlo);
    } else {
      incoming_edge_count[hlo] =
          std::set<HloInstruction*>(hlo->operands().begin(),
                                    hlo->operands().end())
              .size();
    }
  }
  
  //=============================================================
  std::vector<HloInstruction*> fwd_ops;
  std::map<int,std::map<int, HloInstruction*, std::greater<int> >> fwd_overlap_wgrad_ops;
  std::vector<HloInstruction*> dummy_ops;
  std::map<int, std::vector<HloInstruction*>> bwd_overlap_wgrad_ops;

  std::map<int, HloInstruction*, std::greater<int>> f_wgrad_ops;

  for (auto* hlo : computation->instructions()) {
    if (IsForwardOp(*hlo)) {
      fwd_ops.push_back(hlo);  
      //std::cout << "### forward op : " << hlo->metadata().op_name() << "\n";
    } else if (OverlapForward(*hlo)) {
      int target_forward_id = GetOverlapTargetLayerId(*hlo);
      auto it = fwd_overlap_wgrad_ops.find(target_forward_id);
      if (it == fwd_overlap_wgrad_ops.end()) {
        //std::cout << "create wgrad ops!!!!!!!!!!!!!!!! target  : "<< target_forward_id << "\n";
        std::map<int, HloInstruction*, std::greater<int>> wgrad_ops;
        fwd_overlap_wgrad_ops[target_forward_id] = std::move(wgrad_ops);
      }

      int mylayer_id = GetLayerId(*hlo);
      //std::cout << "### overlap forward op : " << hlo->metadata().op_name() << "my layer : " << mylayer_id << " target : " << target_forward_id << "\n";
      HloInstruction* update_op = hlo->users()[0]->users()[0];
      //std::cout << "### overlap forward update op  : " << update_op->metadata().op_name() << "\n";

      //fwd_overlap_wgrad_ops[target_forward_id].push_back(hlo);
      //std::map<int,  HloInstruction*, std::greater<int>> wgrad_ops = fwd_overlap_wgrad_ops[target_forward_id];
      f_wgrad_ops[mylayer_id] = hlo;

    } else if (IsDummyForOverlapForward(*hlo)) {
      HloInstruction* dummy_op = hlo->users()[0];
      HloInstruction* update_op = hlo->users()[0]->users()[0];
      dummy_op->RemoveUser(update_op);
      incoming_edge_count[update_op] -= 1;

      //std::cout << "### dummy for Overalp Forward op : " << hlo->metadata().op_name() << " " << hlo->name()  << "\n";
      //std::cout << "### update OP : " << update_op->metadata().op_name() << " " << update_op->name() <<"\n";
      dummy_ops.push_back(hlo);
    } else if (OverlapOutputGrad(*hlo)) {
      int target_output_grad_id = 0;
      std::string op_name = hlo->metadata().op_name();
      if (op_name.find("Depthwise") != std::string::npos) {
        target_output_grad_id = GetOverlapTargetLayerId(*hlo);
      } else {
        target_output_grad_id = GetOverlapTargetLayerId(*hlo)+1;
      }

      auto it = bwd_overlap_wgrad_ops.find(target_output_grad_id);
      if (it == bwd_overlap_wgrad_ops.end()) {
        std::vector<HloInstruction*> wgrad_ops;
        bwd_overlap_wgrad_ops[target_output_grad_id] = wgrad_ops;
      }

      //std::cout << "### OverlapOutputgrad op : " << hlo->metadata().op_name() << "\n";
      bwd_overlap_wgrad_ops[target_output_grad_id].push_back(hlo);
    }
  }

  while (!queue.empty()) {
    HloInstruction* x = queue.front();
    queue.pop_front();

    if (IsNormalOp(*x)) {
      launch_order->push_back(x);
      PropagateNextNodes(x);
    }

    if (IsForwardOp(*x)) {
      int layer_id = GetLayerId(*x);
      auto it = fwd_overlap_wgrad_ops.find(layer_id);
      if (it != fwd_overlap_wgrad_ops.end()) {
        for (auto iter = f_wgrad_ops.begin(); iter != f_wgrad_ops.end(); iter++) { 
            HloInstruction* wgrad_op = iter->second;
            launch_order->push_back(wgrad_op);

            for (HloInstruction* y : wgrad_op->users()) {
              --incoming_edge_count[y];
              if (incoming_edge_count[y] == 0) {
                launch_order->push_back(y);
                for (HloInstruction* yy : y->users()) {
                    --incoming_edge_count[yy];
                    if (incoming_edge_count[yy] == 0) {
                        launch_order->push_back(yy);
                        PropagateNextNodes(yy);
                    }
                }
              }
            }
        }   
      }
    } else if (IsOutputGradOp(*x)) {
      int layer_id = GetLayerId(*x);
      for (auto* wgrad_op : bwd_overlap_wgrad_ops[layer_id]) {
        launch_order->push_back(wgrad_op);
        PropagateNextNodes(wgrad_op);
        wgrad_op->AppendOperand(x);
        incoming_edge_count[wgrad_op] += 1;
      }  
    }
  }

  for (auto* dummy_op : dummy_ops) {
    launch_order->push_back(dummy_op);
    PropagateNextNodes(dummy_op);
  }

  while (!queue.empty()) {
    HloInstruction* x = queue.front();
    queue.pop_front();
    launch_order->push_back(x);
    for (HloInstruction* y : x->users()) {
      --incoming_edge_count[y];
      if (incoming_edge_count[y] == 0) {
        queue.push_back(y);
      }
    }
  }

  /*
  std::cout << "####### launch order ############## \n";
  for (auto* hlo : *launch_order) {
      std::cout << "op : " << hlo->metadata().op_name() << " " << hlo->name()  << "\n";
  }
  */
}

}  // end namespace

GpuHloSchedule::GpuHloSchedule() {
}

/* static */
StatusOr<std::unique_ptr<GpuHloSchedule>> GpuHloSchedule::Build(
    const HloModule& module, const StreamAssignment& stream_assignment,
    int64 pointer_size) {
  std::unique_ptr<GpuHloSchedule> schedule(new GpuHloSchedule);

  bool do_ooo_backprop = false;
  const char* cstr_do_ooo_backprop = std::getenv("DO_OOO_BACKPROP");
  std::string str_do_ooo_backprop(cstr_do_ooo_backprop ? cstr_do_ooo_backprop : "");
  if (!str_do_ooo_backprop.empty()) {
    do_ooo_backprop = true;
  }

  // Initialize thunk_launch_order_, the total order of thunk launches.
  HloComputation* entry_computation = module.entry_computation();
  if (do_ooo_backprop) {
    MakeOOOLaunchOrder(entry_computation, &schedule->thunk_launch_order_);  
  } else {
    // BFS tends to increase concurrency, but also increases memory usage.
    BFSLaunchOrder(entry_computation, &schedule->thunk_launch_order_);
  }

  schedule->hlo_ordering_ = absl::make_unique<GpuHloOrdering>(
      &module, stream_assignment, schedule->thunk_launch_order_);

  return std::move(schedule);
}

}  // namespace gpu
}  // namespace xla
