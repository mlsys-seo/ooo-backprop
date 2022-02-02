from .utils import *

class DeviceAllocatorManager:
  def __init__ (self,
                gpu_size,
                cluster_size,
                mini_batch_size,
                micro_batch_size,
                layer_size,
                modulo_batch,
                schedule_type,
                is_pretrain=False):

    self.gpu_size = gpu_size
    self.cluster_size = cluster_size
    self.total_gpu_size = int(gpu_size * cluster_size)
    self.mini_batch_size = mini_batch_size
    self.micro_batch_size = micro_batch_size
    self.layer_size = layer_size
    self.modulo_batch = modulo_batch
    self.schedule_type = schedule_type
    self.is_pretrain=is_pretrain

    if MODULO not in schedule_type:
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
    if self.is_pretrain and GPIPE not in self.schedule_type:
      return 'task:%d/GPU:%d' % (self.cluster_size-1, self.get_gpu_seq_idx(self.gpu_size-1))
    return 'task:0/GPU:0'
    
  def get_pooler_device(self):
    return 'task:%d/GPU:%d' % (self.cluster_size-1, self.get_gpu_seq_idx(self.gpu_size-1))
  
  def get_loss_device(self):
    return 'task:%d/GPU:%d' % (self.cluster_size-1, self.get_gpu_seq_idx(self.gpu_size-1))


  def get_encoder_layer_device(self, layer_idx, micro_batch_idx):
    if MODULO in self.schedule_type:
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