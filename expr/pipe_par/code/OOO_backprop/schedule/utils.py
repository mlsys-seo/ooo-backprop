VIRTUAL_LAYER_PREFIX="ST_"
VIRTUAL_LAYER_SUFFIX="_ED"

MICRO_BATCH_PREFIX="train_"
MICRO_BATCH_SUFFIX="th/"

MODEL_FIRST_OP="bert/embeddings/FIRST_OP/dim"
FORWARD_FIRST_OP="/attention/self/query/MatMul/ReadVariableOp"
BACKWARD_FIRST_OP="/output/layer_norm/layer_normalization"
SEND="AddN"
ATTENTION="/attention"
MUL="Mul_1"
MUL_GRAD="/mul_2_grad/Mul"

GRADIENTS="gradients"
OUTPUT_GRAD_NAME="OUT_GRAD"
WEIGHT_GRAD_NAME="WEIGHT_GRAD"
LOSS='loss'
POOLER='pooler'
EMBEDDING='embeddings'

MODULO = 'mod'
FASTFORWARD = 'fast'
GPIPE = 'gpipe'

PUSH = 'push'

def get_idx_util(op_name, PREFIX, SUFFIX):
  if PREFIX not in op_name:
      return -1

  POSTFIX_SIZE = len(PREFIX)
  prefix_idx = op_name.find(PREFIX)
  tmp_name = op_name[prefix_idx:]
  suffix_idx = tmp_name.find(SUFFIX)
  return int(tmp_name[POSTFIX_SIZE:suffix_idx])

def get_virtual_layer_idx(op_name):
  return get_idx_util(op_name, VIRTUAL_LAYER_PREFIX, VIRTUAL_LAYER_SUFFIX)

def get_micro_batch_idx(op_name):
  return get_idx_util(op_name, MICRO_BATCH_PREFIX, MICRO_BATCH_SUFFIX)

def set_dependency_util(before_op, next_op):
  next_op._add_control_input( before_op )

