class ModelOps:
  def __init__(self, forward_last_op_per_virtual_layers):
    self.model_first_op = None
    self.forward_first_op_per_virtual_layers = {}
    self.forward_last_op_per_virtual_layers = forward_last_op_per_virtual_layers
    self.backward_first_op_per_virtual_layers = {}
    self.output_gradient_ops = {} 
    self.weight_gradient_ops = {}
    self.backward_last_op_per_virtual_layers = {}
    self.backward_send_op_per_virtual_layers = {}

    
  def set_model_first_op(self, op):
    self.model_first_op = op

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