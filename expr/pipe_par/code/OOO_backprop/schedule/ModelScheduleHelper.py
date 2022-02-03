from .ModelOps import ModelOps
from .utils import *

class ModelScheduleHelper:
  def __init__ (self, 
                graph, 
                gpu_size, 
                layer_size,
                modulo_batch,
                micro_batch_size,
                forward_last_op_per_virtual_layers, 
                schedule_type = GPIPE):

    self.micro_models = []
    for micro_batch_idx in range(micro_batch_size):
      self.micro_models.append(ModelOps(forward_last_op_per_virtual_layers[micro_batch_idx]))

    self.graph = graph
    self.gpu_size = gpu_size
    self.micro_batch_size = micro_batch_size
    self.virtual_layers_size = int(layer_size / modulo_batch) if MODULO in schedule_type else self.gpu_size

    self.schedule_type = schedule_type

    self.push = self.micro_batch_size > self.gpu_size and PUSH in self.schedule_type

    self.micro_batch_cap = self.gpu_size if self.push else self.micro_batch_size
    self.remainder_micro_batch = self.micro_batch_size - self.micro_batch_cap if self.micro_batch_size > self.micro_batch_cap else 0


  def schedule(self):
    self._prepare_forward_process()
    self._prepare_backward_process()

    self._schedule_forward_process()
    if GPIPE not in self.schedule_type:
      self._schedule_backward_process()
    else :
      self._schedule_gpipe_last_stage()

    if self.push:
      self._schedule_microbatch_cap_fastforward()

  def _prepare_forward_process(self):
    self._find_and_set_model_first_op()
    self._find_and_set_forward_first_op_per_virtual_layers()
    self._find_and_set_forward_last_op()
    self._find_and_set_backward_first_op_per_virtual_layers()
    self._find_and_set_backward_first_op()

  def _prepare_backward_process(self):
    self._find_and_set_weight_gradient_ops()
    self._find_and_set_backward_send_op_per_virtual_layers()



  def _schedule_forward_process(self):
    self._schedule_forward_first_virtual_layers()
    self._schedule_forward_second_virtual_layers()
    if GPIPE not in self.schedule_type:
      self._schedule_forward_second_last_virtual_layers()

  def _schedule_race_condition(self):
    self._schedule_micro_batch_race_condition()
    if MODULO in self.schedule_type:
      self._schedule_virtual_layers_race_condition()

  def _schedule_backward_process(self):
    self._schedule_ooo_backpropagation()
    self._schedule_race_condition()


  def _is_model_first_op(self, op):
    return MODEL_FIRST_OP in op.name

  def _is_forward_first_op_per_virtual_layers(self, op):
    return FORWARD_FIRST_OP in op.name and GRADIENTS not in op.name

  def _is_forward_last_op(self, op):
    return FORWARD_LAST_OP in op.name and GRADIENTS not in op.name

  def _is_backward_first_op_per_virtual_layers(self, op):
    return GRADIENTS in op.name and MUL_GRAD in op.name and MUL not in op.name and \
           ATTENTION not in op.name and BACKWARD_FIRST_OP in op.name

  def _is_backward_first_op(self, op):
    return FORWARD_LAST_OP in op.name and RESHAPE in op.name and GRADIENTS in op.name

  def _is_output_gradient_ops(self, op):
    return OUTPUT_GRAD_NAME in op.name

  def _is_weight_gradient_ops(self, op):
    return WEIGHT_GRAD_NAME in op.name

  def _is_backward_send_op_per_virtual_layers(self, op, map, key):
    return SEND in op.name and any( [input_ops.op is map[key] for input_ops in op.inputs] )


  def set_model_first_op(self, op):
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_model_first_op(op)

  def set_forward_first_op_per_virtual_layers(self, op):
    virtual_layers_idx = get_virtual_layer_idx(op.name)
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_forward_first_op_per_virtual_layers(op, virtual_layers_idx)

  def set_forward_last_op(self, op):
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_forward_last_op(op)

  def set_backward_first_op_per_virtual_layers(self, op):
    virtual_layers_idx = get_virtual_layer_idx(op.name)
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_backward_first_op_per_virtual_layers(op, virtual_layers_idx)

  def set_backward_first_op(self, op):
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_backward_first_op(op)

  def set_output_gradient_ops(self, op):
    virtual_layers_idx = get_virtual_layer_idx(op.name)
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_output_gradient_ops(op, virtual_layers_idx)

  def set_weight_gradient_ops(self, op):
    virtual_layers_idx = get_virtual_layer_idx(op.name)
    if virtual_layers_idx is -1:
      virtual_layers_idx = self.virtual_layers_size-1 if LOSS in op.name or POOLER in op.name else -1
    micro_batch_idx = get_micro_batch_idx(op.name)
    self.micro_models[micro_batch_idx].set_weight_gradient_ops(op, virtual_layers_idx)

  def set_backward_last_op_per_virtual_layers(self, op, micro_batch_idx, virtual_layers_idx):
    self.micro_models[micro_batch_idx].set_backward_last_op_per_virtual_layers(op, virtual_layers_idx)

  def set_backward_send_op_per_virtual_layers(self, op, micro_batch_idx, virtual_layers_idx):
    self.micro_models[micro_batch_idx].set_backward_send_op_per_virtual_layers(op, virtual_layers_idx)



  def _find_and_set_model_first_op(self):
    for op in self.graph.get_operations():
      if self._is_model_first_op(op):
        self.set_model_first_op(op)

  def _find_and_set_forward_first_op_per_virtual_layers(self):
    for op in self.graph.get_operations():
      if self._is_forward_first_op_per_virtual_layers(op):
        self.set_forward_first_op_per_virtual_layers(op)

  def _find_and_set_forward_last_op(self):
    for op in self.graph.get_operations():
      if self._is_forward_last_op(op):
        self.set_forward_last_op(op)

  def _find_and_set_backward_first_op_per_virtual_layers(self):  
    for op in self.graph.get_operations():
      if self._is_backward_first_op_per_virtual_layers(op):
        self.set_backward_first_op_per_virtual_layers(op)

  def _find_and_set_backward_first_op(self):
    for op in self.graph.get_operations():
      if self._is_backward_first_op(op):
        self.set_backward_first_op(op)

  def _find_and_set_weight_gradient_ops(self):
    for op in self.graph.get_operations():
      if self._is_weight_gradient_ops(op):
        self.set_weight_gradient_ops(op)

  def _find_and_set_output_gradient_ops(self):
    for op in self.graph.get_operations():
      if self._is_output_gradient_ops(op):
        self.set_output_gradient_ops(op)

  def _find_and_set_backward_last_op_per_virtual_layers(self):
    self._find_and_set_output_gradient_ops()
    for micro_batch_idx in range(self.micro_batch_size):
      map = self.micro_models[micro_batch_idx].get_output_gradient_ops()
      for virtual_layers_idx in range(self.virtual_layers_size):
        op = map[virtual_layers_idx][-1]
        self.set_backward_last_op_per_virtual_layers(op, micro_batch_idx, virtual_layers_idx)

  def _find_and_set_backward_send_op_per_virtual_layers(self):
    self._find_and_set_backward_last_op_per_virtual_layers()
    for micro_batch_idx in range(self.micro_batch_size):
      map = self.micro_models[micro_batch_idx].get_backward_last_op_per_virtual_layers()
      for op in self.graph.get_operations():
        for virtual_layers_idx in range(self.virtual_layers_size):
          if self._is_backward_send_op_per_virtual_layers(op, map, virtual_layers_idx):
            self.set_backward_send_op_per_virtual_layers(op, micro_batch_idx, virtual_layers_idx)


  def _schedule_forward_first_virtual_layers(self):
    for micro_batch_idx in range(self.micro_batch_size-1):
      model_first_op = self.micro_models[micro_batch_idx+1].get_model_first_op()
      forward_last_op_per_virtual_layers = self.micro_models[micro_batch_idx].get_forward_last_op_per_virtual_layers()
      set_dependency_util(forward_last_op_per_virtual_layers[0].op, model_first_op)

  def _schedule_forward_second_virtual_layers(self):
    for micro_batch_idx in range(self.micro_batch_size-1):
      forward_first_op_per_virtual_layers = self.micro_models[micro_batch_idx+1].get_forward_first_op_per_virtual_layers()
      forward_last_op_per_virtual_layers = self.micro_models[micro_batch_idx].get_forward_last_op_per_virtual_layers()
      set_dependency_util(forward_last_op_per_virtual_layers[1].op, forward_first_op_per_virtual_layers[1])

  def _schedule_forward_second_last_virtual_layers(self):
    forward_last_op_per_virtual_layers = self.micro_models[self.micro_batch_cap-1].get_forward_last_op_per_virtual_layers()
    backward_first_op_per_virtual_layers = self.micro_models[0].get_backward_first_op_per_virtual_layers()
    set_dependency_util( forward_last_op_per_virtual_layers[self.virtual_layers_size-2].op, 
                              backward_first_op_per_virtual_layers[self.virtual_layers_size-2])

  def _schedule_gpipe_last_stage(self):
    forward_last_op = self.micro_models[self.micro_batch_size-1].get_forward_last_op()
    for micro_batch_idx in range(self.micro_batch_size):
      backward_first_op = self.micro_models[micro_batch_idx].get_backward_first_op()
      set_dependency_util( forward_last_op, backward_first_op )

    for micro_batch_idx in range(self.micro_batch_size-1):
      backward_stage_last_ops = self.micro_models[micro_batch_idx].get_backward_send_op_per_virtual_layers()
      backward_stage_last_op = backward_stage_last_ops[self.virtual_layers_size-1]
      backward_last_stage_first_op = self.micro_models[micro_batch_idx+1].get_backward_first_op()
      set_dependency_util( backward_stage_last_op, backward_last_stage_first_op )
    

  def _schedule_microbatch_cap_fastforward(self):
    for micro_batch_idx in range(self.remainder_micro_batch):
      for virtual_layer_idx in range(self.virtual_layers_size-1):
        weight_grad_map = self.micro_models[micro_batch_idx+virtual_layer_idx].get_weight_gradient_ops()
        weight_grad_ops = weight_grad_map[virtual_layer_idx]
        if virtual_layer_idx is 0:
          first_forward_op = self.micro_models[self.micro_batch_cap+micro_batch_idx].get_model_first_op()
        else :
          first_forward_ops = self.micro_models[self.micro_batch_cap+micro_batch_idx].get_forward_first_op_per_virtual_layers()
          first_forward_op = first_forward_ops[virtual_layer_idx]
        
        for weight_gradient_op in weight_grad_ops:
          set_dependency_util(weight_gradient_op, first_forward_op)
      

  def _schedule_ooo_backpropagation(self):
    for micro_batch_idx in range(self.micro_batch_size):
      weight_grad_map = self.micro_models[micro_batch_idx].get_weight_gradient_ops()
      backward_send_ops = self.micro_models[micro_batch_idx].get_backward_send_op_per_virtual_layers()

      if self.virtual_layers_size is self.gpu_size:
        for virtual_layers_idx in range(self.virtual_layers_size):
          if (virtual_layers_idx+1) % self.gpu_size:
            weight_grad_ops = weight_grad_map[virtual_layers_idx]
            send_grad_op = backward_send_ops[virtual_layers_idx]
            for weight_grad_op in weight_grad_ops:
              set_dependency_util(send_grad_op, weight_grad_op)
          else:
            backward_send_last_op = self.micro_models[self.micro_batch_size-1].get_backward_send_op_per_virtual_layers()
            weight_grad_ops = weight_grad_map[virtual_layers_idx]
            send_grad_op = backward_send_last_op[self.virtual_layers_size-1]
            for weight_grad_op in weight_grad_ops:
              set_dependency_util(send_grad_op, weight_grad_op)
      else:
        for virtual_layers_idx in range(self.virtual_layers_size-1):
          if (virtual_layers_idx+1) % self.gpu_size:
            weight_grad_ops = weight_grad_map[virtual_layers_idx]
          else :
            weight_grad_ops = weight_grad_map[virtual_layers_idx+self.gpu_size]
          send_grad_op = backward_send_ops[virtual_layers_idx]
          for weight_grad_op in weight_grad_ops:
            set_dependency_util(send_grad_op, weight_grad_op)

        backward_send_last_op = self.micro_models[self.micro_batch_size-1].get_backward_send_op_per_virtual_layers()
        send_grad_op = backward_send_last_op[self.gpu_size-1]
        weight_grad_ops = weight_grad_map[self.gpu_size-1]
        for weight_grad_op in weight_grad_ops:
          set_dependency_util(send_grad_op, weight_grad_op)



  def _schedule_micro_batch_race_condition(self):
    for micro_batch_idx in range(self.micro_batch_size-1):
      weight_grad_map = self.micro_models[micro_batch_idx].get_weight_gradient_ops()
      backward_first_ops = self.micro_models[micro_batch_idx+1].get_backward_first_op_per_virtual_layers()

      for virtual_layers_idx in range(self.virtual_layers_size-1):
        vlayer_idx = virtual_layers_idx if (virtual_layers_idx+1) % self.gpu_size else virtual_layers_idx+self.gpu_size
        weight_grad_ops = weight_grad_map[vlayer_idx]
        backward_first_op = backward_first_ops[virtual_layers_idx]
        for weight_grad_op in weight_grad_ops:
          set_dependency_util(weight_grad_op, backward_first_op)


  def _schedule_virtual_layers_race_condition(self):
    weight_grad_map = self.micro_models[self.micro_batch_size-1].get_weight_gradient_ops()
    backward_first_ops = self.micro_models[0].get_backward_first_op_per_virtual_layers()
    for virtual_layers_idx in range(self.virtual_layers_size-self.gpu_size-1):
      vlayer_idx = virtual_layers_idx+self.gpu_size if (virtual_layers_idx+1) % self.gpu_size else virtual_layers_idx+2*self.gpu_size
      weight_grad_ops = weight_grad_map[vlayer_idx]
      backward_first_op = backward_first_ops[virtual_layers_idx]
      for weight_grad_op in weight_grad_ops:
        set_dependency_util(weight_grad_op, backward_first_op)

    backward_first_op = backward_first_ops[self.virtual_layers_size-self.gpu_size-1]
    backward_send_ops = self.micro_models[self.micro_batch_size-1].get_backward_send_op_per_virtual_layers()
    backward_send_op = backward_send_ops[self.virtual_layers_size-1]
    set_dependency_util(backward_send_op, backward_first_op)