class ModelOps:
  def __init__(self, forward_last_op_per_virtual_layers):
    self.model_first_op = None
    self.model_last_op = None
    self.forward_first_op_per_virtual_layers = {}
    self.forward_last_op_per_virtual_layers = forward_last_op_per_virtual_layers
    self.backward_first_op_per_virtual_layers = {}
    self.output_gradient_ops = {} 
    self.weight_gradient_ops = {}
    self.backward_last_op_per_virtual_layers = {}
    self.backward_send_op_per_virtual_layers = {}

    
  def set_model_first_op(self, op):
    self.model_first_op = op

  def set_model_last_op(self, op):
    self.model_last_op = op

  def set_forward_first_op_per_virtual_layers(self, op, key):
    if key not in self.forward_first_op_per_virtual_layers:
      self.forward_first_op_per_virtual_layers[key] = op

  def set_backward_first_op_per_virtual_layers(self, op, key):
    if key not in self.backward_first_op_per_virtual_layers:
      self.backward_first_op_per_virtual_layers[key] = op

  def set_output_gradient_ops(self, op, key):
    if key not in self.output_gradient_ops:
      self.output_gradient_ops[key] = []
    self.output_gradient_ops[key].append(op)

  def set_weight_gradient_ops(self, op, key):
    if key not in self.weight_gradient_ops:
      self.weight_gradient_ops[key] = []
    self.weight_gradient_ops[key].append(op)

  def set_backward_last_op_per_virtual_layers(self, op, key):
    self.backward_last_op_per_virtual_layers[key] = op

  def set_backward_send_op_per_virtual_layers(self, op, key):
    self.backward_send_op_per_virtual_layers[key] = op


  def get_model_first_op(self):
    return self.model_first_op

  def get_model_last_op(self):
    return self.model_last_op

  def get_forward_first_op_per_virtual_layers(self):
    return self.forward_first_op_per_virtual_layers

  def get_forward_last_op_per_virtual_layers(self):
    return self.forward_last_op_per_virtual_layers

  def get_backward_first_op_per_virtual_layers(self):
    return self.backward_first_op_per_virtual_layers

  def get_output_gradient_ops(self):
    return self.output_gradient_ops

  def get_weight_gradient_ops(self):
    return self.weight_gradient_ops

  def get_backward_last_op_per_virtual_layers(self):
    return self.backward_last_op_per_virtual_layers

  def get_backward_send_op_per_virtual_layers(self):
    return self.backward_send_op_per_virtual_layers


class DeviceAllocatorManager:
  def __init__ (self,
                gpu_size,
                cluster_size,
                mini_batch_size,
                micro_batch_size,
                layer_size,
                modulo_batch=1,
                schedule_type='gpipe'):

    self.gpu_size = gpu_size
    self.cluster_size = cluster_size
    self.total_gpu_size = int(gpu_size * cluster_size)
    self.mini_batch_size = mini_batch_size
    self.micro_batch_size = micro_batch_size
    self.layer_size = layer_size
    self.modulo_batch = modulo_batch
    self.schedule_type = schedule_type

    if 'mod' not in schedule_type:
      self.offset = self.layer_size % self.total_gpu_size
      self.remainder = []
      for micro_batch_idx in range(self.micro_batch_size):
        self.remainder.append(self.offset)
      self.quotient = int(self.layer_size / self.total_gpu_size)
      self.offset = self.layer_size % self.total_gpu_size

    self.gpu_seq = [0, 1, 2, 3, 4, 5, 6, 7] if 'gpipe' in self.schedule_type else [0, 3, 2, 1, 5, 6, 7, 4]

    self.before_gpu_idx = []
    self.v_layer_num = []
    for _ in range(self.micro_batch_size):
      self.before_gpu_idx.append(0)
      self.v_layer_num.append(0)


  def get_gpu_seq_idx(self, gpu_idx):
    return self.gpu_seq[gpu_idx]


  def get_embedding_table_device(self):
    if 'gpipe' not in self.schedule_type:
      return '/task:0/GPU:0'
    # return 'task:%d/GPU:%d' % (self.cluster_size-1, self.get_gpu_seq_idx(self.gpu_size-1))
    return 'task:0/GPU:0'

  def get_embedding_matmul_device(self):
    if 'gpipe' not in self.schedule_type:
      return '/task:0/GPU:0'
    return 'task:%d/GPU:%d' % (self.cluster_size-1, self.get_gpu_seq_idx(self.gpu_size-1))
  

  def get_encoder_layer_device(self, layer_idx, micro_batch_idx):
    if 'mod' in self.schedule_type:
      return self.get_gpu_idx_modulo(layer_idx, micro_batch_idx)
    return self.get_gpu_idx(layer_idx, micro_batch_idx)


  def get_gpu_idx(self, layer_idx, micro_batch_idx):
    if self.remainder[micro_batch_idx]:
      gpu_idx = int((layer_idx+1) / (self.quotient+1))
    else :
      gpu_idx = int((layer_idx-self.offset) / self.quotient)

    if (gpu_idx != self.before_gpu_idx[micro_batch_idx] and self.remainder[micro_batch_idx]):
      self.remainder[micro_batch_idx] -= 1

    new_forward_virtual_layer = False
    if gpu_idx is not self.before_gpu_idx[micro_batch_idx]:
      new_forward_virtual_layer = True
      self.v_layer_num[micro_batch_idx] += 1

    self.before_gpu_idx[micro_batch_idx] = gpu_idx

    cluster = gpu_idx // self.gpu_size
    gpu = self.get_gpu_seq_idx(gpu_idx % self.gpu_size)
    device_name = 'task:%d/GPU:%d' % (cluster, gpu)

    return new_forward_virtual_layer, device_name
      

  def get_gpu_idx_modulo(self, layer_idx, micro_batch_idx):
    gpu_idx = int(layer_idx / self.modulo_batch) % self.total_gpu_size
    new_forward_virtual_layer = False 
    if gpu_idx is not self.before_gpu_idx[micro_batch_idx]:
      new_forward_virtual_layer = True
      self.v_layer_num[micro_batch_idx] += 1

    self.before_gpu_idx[micro_batch_idx] = gpu_idx

    cluster = gpu_idx // self.gpu_size
    gpu = self.get_gpu_seq_idx(gpu_idx % self.gpu_size)
    device_name = 'task:%d/GPU:%d' % (cluster, gpu)

    return new_forward_virtual_layer, device_name

  def get_virtual_layer_number(self, micro_batch_idx):
    return self.v_layer_num[micro_batch_idx]

  def get_virtual_layer_name(self, micro_batch_idx):
    return 'virtual_layer_ST_%d_ED' % self.v_layer_num[micro_batch_idx]





class ModelScheduleHelper:
  def __init__ (self, 
                graph, 
                gpu_size, 
                layer_size,
                modulo_batch,
                micro_batch_size,
                forward_last_op_per_virtual_layers, 
                schedule_type = 'gpipe'):

    self.micro_models = []
    for micro_batch_idx in range(micro_batch_size):
      self.micro_models.append(ModelOps(forward_last_op_per_virtual_layers[micro_batch_idx]))

    self.graph = graph
    self.gpu_size = gpu_size
    self.micro_batch_size = micro_batch_size
    self.virtual_layers_size = int(layer_size / modulo_batch) if 'mod' in schedule_type else self.gpu_size

    self.schedule_type = schedule_type

    self.micro_batch_cap = self.gpu_size if 'fast' in self.schedule_type and 'push' in self.schedule_type else self.micro_batch_size
    self.remainder_micro_batch = self.micro_batch_size - self.micro_batch_cap if self.micro_batch_size > self.micro_batch_cap else 0


  def schedule(self):
    self._prepare_forward_process()
    self._schedule_forward_process()

    if 'gpipe' not in self.schedule_type:
      self._prepare_backward_process()
      self._schedule_backward_process()

    if 'push' in self.schedule_type:
      self._schedule_microbatch_cap_fastforward()

  def _prepare_forward_process(self):
    self._find_and_set_model_first_op()
    self._find_and_set_forward_first_op_per_virtual_layers()
    self._find_and_set_backward_first_op_per_virtual_layers()

  def _prepare_backward_process(self):
    self._find_and_set_weight_gradient_ops()
    self._find_and_set_backward_send_op_per_virtual_layers()



  def _schedule_forward_process(self):
    self._schedule_forward_first_virtual_layers()
    self._schedule_forward_second_virtual_layers() 
    self._schedule_forward_second_last_virtual_layers()


  def _schedule_race_condition(self):
    self._schedule_micro_batch_race_condition()
    self._schedule_virtual_layers_race_condition()

  def _schedule_backward_process(self):
    self._schedule_ooo_backpropagation()
    self._schedule_race_condition()


  #TODO ST_, _ED
  def get_idx_util(self, op_name, PREFIX="ST_", SUFFIX="_ED"):
    if PREFIX not in op_name:
        return -1

    POSTFIX_SIZE = len(PREFIX)
    prefix_idx = op_name.find(PREFIX)
    tmp_name = op_name[prefix_idx:]
    suffix_idx = tmp_name.find(SUFFIX)
    return int(tmp_name[POSTFIX_SIZE:suffix_idx])

  def set_dependency_util(self, before_op, next_op):
    # print(before_op.name, "->", next_op.name)
    next_op._add_control_input( before_op )



  def _is_model_first_op(self, op):
    return "bert/embeddings/FIRST_OP/dim" in op.name
    
  def _is_model_last_op(self, op):
    return "gradients" in op.name and "/embeddings/MatMul_grad/OUT_GRAD" in op.name

  def _is_forward_first_op_per_virtual_layers(self, op):
    return "gradients" not in op.name and \
           "/attention/self/query/MatMul/ReadVariableOp" in op.name

  def _is_backward_first_op_per_virtual_layers(self, op):
    return "gradients" in op.name and \
           "Mul_1" not in op.name and \
           "/mul_2_grad/Mul" in op.name and \
           "/output/layer_norm/layer_normalization" in op.name

  def _is_output_gradient_ops(self, op):
    return "OUT_GRAD" in op.name

  def _is_weight_gradient_ops(self, op):
    return "WEIGHT_GRAD" in op.name

  def _is_backward_send_op_per_virtual_layers(self, op, map, key):
    return "AddN_" in op.name and \
           any( [input_ops.op is map[key] for input_ops in op.inputs] )


  def set_model_first_op(self, op):
    micro_batch_idx = self.get_idx_util(op.name, "train_", "th/")
    self.micro_models[micro_batch_idx].set_model_first_op(op)

  def set_model_last_op(self, op):
    micro_batch_idx = self.get_idx_util(op.name, "train_", "th/")
    self.micro_models[micro_batch_idx].set_model_last_op(op)

  def set_forward_first_op_per_virtual_layers(self, op):
    virtual_layers_idx = self.get_idx_util(op.name)
    micro_batch_idx = self.get_idx_util(op.name, "train_", "th/")
    self.micro_models[micro_batch_idx].set_forward_first_op_per_virtual_layers(op, virtual_layers_idx)

  def set_backward_first_op_per_virtual_layers(self, op):
    virtual_layers_idx = self.get_idx_util(op.name)
    micro_batch_idx = self.get_idx_util(op.name, "train_", "th/")
    self.micro_models[micro_batch_idx].set_backward_first_op_per_virtual_layers(op, virtual_layers_idx)

  def set_output_gradient_ops(self, op):
    virtual_layers_idx = self.get_idx_util(op.name)
    micro_batch_idx = self.get_idx_util(op.name, "train_", "th/")
    self.micro_models[micro_batch_idx].set_output_gradient_ops(op, virtual_layers_idx)

  def set_weight_gradient_ops(self, op):
    virtual_layers_idx = self.get_idx_util(op.name)
    micro_batch_idx = self.get_idx_util(op.name, "train_", "th/")
    self.micro_models[micro_batch_idx].set_weight_gradient_ops(op, virtual_layers_idx)

  def set_backward_last_op_per_virtual_layers(self, op, micro_batch_idx, virtual_layers_idx):
    self.micro_models[micro_batch_idx].set_backward_last_op_per_virtual_layers(op, virtual_layers_idx)

  def set_backward_send_op_per_virtual_layers(self, op, micro_batch_idx, virtual_layers_idx):
    self.micro_models[micro_batch_idx].set_backward_send_op_per_virtual_layers(op, virtual_layers_idx)



  def _find_and_set_model_first_op(self):
    for op in self.graph.get_operations():
      if self._is_model_first_op(op):
        self.set_model_first_op(op)

  def _find_and_set_model_last_op(self):
    for op in self.graph.get_operations():
      if self._is_model_last_op(op):
        print(op.name)
        self.set_model_last_op(op)
        
  def _find_and_set_forward_first_op_per_virtual_layers(self):
    for op in self.graph.get_operations():
      if self._is_forward_first_op_per_virtual_layers(op):
        self.set_forward_first_op_per_virtual_layers(op)

  def _find_and_set_backward_first_op_per_virtual_layers(self):  
    for op in self.graph.get_operations():
      if self._is_backward_first_op_per_virtual_layers(op):
        self.set_backward_first_op_per_virtual_layers(op)

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
      self.set_dependency_util(forward_last_op_per_virtual_layers[0].op, model_first_op)

  def _schedule_forward_second_virtual_layers(self):
    for micro_batch_idx in range(self.micro_batch_size-1):
      forward_first_op_per_virtual_layers = self.micro_models[micro_batch_idx+1].get_forward_first_op_per_virtual_layers()
      forward_last_op_per_virtual_layers = self.micro_models[micro_batch_idx].get_forward_last_op_per_virtual_layers()
      self.set_dependency_util(forward_last_op_per_virtual_layers[1].op, forward_first_op_per_virtual_layers[1])

  def _schedule_forward_second_last_virtual_layers(self):
    forward_last_op_per_virtual_layers = self.micro_models[self.micro_batch_cap-1].get_forward_last_op_per_virtual_layers()
    backward_first_op_per_virtual_layers = self.micro_models[0].get_backward_first_op_per_virtual_layers()
    self.set_dependency_util( forward_last_op_per_virtual_layers[self.virtual_layers_size-2].op, 
                              backward_first_op_per_virtual_layers[self.virtual_layers_size-2])

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
          self.set_dependency_util(weight_gradient_op, first_forward_op)
      

  def _schedule_ooo_backpropagation(self):
    for micro_batch_idx in range(self.micro_batch_size):
      weight_grad_map = self.micro_models[micro_batch_idx].get_weight_gradient_ops()
      backward_send_ops = self.micro_models[micro_batch_idx].get_backward_send_op_per_virtual_layers()

      for virtual_layers_idx in range(self.virtual_layers_size):
        weight_grad_ops = weight_grad_map[virtual_layers_idx]
        send_grad_op = backward_send_ops[virtual_layers_idx]
        for weight_grad_op in weight_grad_ops:
          self.set_dependency_util(send_grad_op, weight_grad_op)

  def _schedule_micro_batch_race_condition(self):
    for micro_batch_idx in range(self.micro_batch_size-1):
      weight_grad_map = self.micro_models[micro_batch_idx].get_weight_gradient_ops()
      backward_first_ops = self.micro_models[micro_batch_idx+1].get_backward_first_op_per_virtual_layers()

      for virtual_layers_idx in range(self.virtual_layers_size-1):
        weight_grad_ops = weight_grad_map[virtual_layers_idx]
        backward_first_op = backward_first_ops[virtual_layers_idx]
        for weight_grad_op in weight_grad_ops:
          self.set_dependency_util(weight_grad_op, backward_first_op)

  def _schedule_virtual_layers_race_condition(self):
    weight_grad_map = self.micro_models[self.micro_batch_size-1].get_weight_gradient_ops()
    backward_first_ops = self.micro_models[0].get_backward_first_op_per_virtual_layers()
    for virtual_layers_idx in range(self.gpu_size, self.virtual_layers_size-1):
      weight_grad_ops = weight_grad_map[virtual_layers_idx]
      backward_first_op = backward_first_ops[virtual_layers_idx-self.gpu_size]
      for weight_grad_op in weight_grad_ops:
        self.set_dependency_util(weight_grad_op, backward_first_op)

    if 'mod' in self.schedule_type:
      backward_first_op_in_rush = backward_first_ops[self.virtual_layers_size-self.gpu_size-1]
      backward_first_ops = self.micro_models[self.micro_batch_size-1].get_backward_first_op_per_virtual_layers()
      backward_first_op = backward_first_ops[self.virtual_layers_size-2]
      self.set_dependency_util(backward_first_op, backward_first_op_in_rush)

      for micro_batch_idx in range(self.micro_batch_size-1):
        backward_first_ops = self.micro_models[micro_batch_idx].get_backward_first_op_per_virtual_layers()
        backward_first_op = backward_first_ops[self.virtual_layers_size-self.gpu_size-2]
        backward_first_ops_in_rush = self.micro_models[micro_batch_idx+1].get_backward_first_op_per_virtual_layers()
        backward_first_op_in_rush = backward_first_ops_in_rush[self.virtual_layers_size-self.gpu_size-1]
        self.set_dependency_util(backward_first_op, backward_first_op_in_rush)

