class ModelOps:
  def __init__(self, forward_last_op_per_virtual_layers):
    # [E][012]                                      [OOO][E][WWW]
    #         [345]                            [OOO][WWW]
    #              [678]                  [OOO][WWW]
    #                   [9ab][EL][LE][OOO][WWW]

    self.model_first_op = None                          # Model's very first operation. (Embedding Lookup)
    self.forward_first_op_per_virtual_layers = {}       # First operation per each Encoder layer batch.
    self.forward_last_op_per_virtual_layers = \
              forward_last_op_per_virtual_layers        # Last operation per each Encoder layer batch.
    self.forward_last_op = None                         # Model's very last operation. (Loss)
    self.backward_first_op = None                       # Backpropagation's very first operation. (Loss)
    self.backward_first_op_per_virtual_layers = {}      # First operation per each Backpropagation Encoder layer batch.
    self.output_gradient_ops = {}                       # Output Gradients grouped by Backpropagation Encoder layer batch.
    self.weight_gradient_ops = {}                       # Weight Gradients grouped by Backpropagation Encoder layer batch.
    self.backward_last_op_per_virtual_layers = {}       # Last operation per each Backpropagation Encoder layer batch.
    self.backward_send_op_per_virtual_layers = {}       # Communication operation sending the last Output Gradients per each Backpropagation Encoder layer batch.

    
  def set_model_first_op(self, op):
    self.model_first_op = op

  def set_forward_first_op_per_virtual_layers(self, op, key):
    if key not in self.forward_first_op_per_virtual_layers:
      self.forward_first_op_per_virtual_layers[key] = op

  def set_forward_last_op(self, op):
    self.forward_last_op = op

  def set_backward_first_op_per_virtual_layers(self, op, key):
    if key not in self.backward_first_op_per_virtual_layers:
      self.backward_first_op_per_virtual_layers[key] = op

  def set_backward_first_op(self, op):
    self.backward_first_op = op

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

  def get_forward_last_op(self):
    return self.forward_last_op

  def get_forward_last_op_per_virtual_layers(self):
    return self.forward_last_op_per_virtual_layers

  def get_backward_first_op(self):
    return self.backward_first_op

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