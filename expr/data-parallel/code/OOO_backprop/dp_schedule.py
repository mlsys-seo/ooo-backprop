import tensorflow as tf
import byteps

from OOO_backprop import get_args
from OOO_backprop import print_log
from OOO_backprop import print_graph

from byteps.tensorflow import comm_terminal
from byteps.tensorflow import polling_data

args = get_args()

def is_async_op(op):
    if args.async_op in op.name:
        return True
    else:
        return False

def is_send_recv_op(op):
    if "Push_Pull" in op.name:
        return True
    else:
        return False

def is_w_grad_op(op):
    if "Conv2DBackpropFilter" in op.name:
        return True
    else:
        return False

def is_conv_op(op_name):
    if "CONV_LAYER_" in op_name:
        return True
    else:
        return False

def encode_sched_map_into_name(layer_name, curr_layer_num=-1): # SK: 이건 models.py로 가도 될듯
    """Marks async_op tag to name of operation which idx < `reverse_first_k`
    
    Parameters
    ----------
        layer_name: str
            Layer Name

        curr_layer_num: int
            Current index of layer
    
    Returns
    -------
        layer_name: str
            Edited layer name
    """
    
    layer_name = f"{layer_name}{curr_layer_num}{args.separater}"

    if not is_conv_op(layer_name):
        return layer_name

    if curr_layer_num < args.reverse_first_k:
        layer_name = layer_name + args.async_op
    
    return layer_name


forward_ops_table = {}

def collect_conv_op(op, layer_num):
    """Collects convolution operations for scheduling
    
    Parameters
    ----------
        op: tf.operation
            Operation object
        layer_num: int
            Current index of layer
    
    Returns
    -------
        None
    """
    forward_ops_table[layer_num] = op


class OOO_ScheduleHelper:
    """Helper Object for adopting OOO schedule.
    
    Methods
    ----------
        schedule_ops: Returns list of tf.Operation
    
    """
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.COMMUNICATION_CONV = 0

    def _remove_w_grad_dependency(self, op):
        control_dependencies = op.control_inputs
        op._remove_all_control_inputs()

        # pop weight grad op
        control_dependencies = [ dep_op for dep_op in control_dependencies if not is_w_grad_op(dep_op)]
        op._add_control_inputs(control_dependencies)

    def _find_forward_op(self, v, key):
        forward_op = forward_ops_table[key]
        return forward_op

    def _get_op_index(self, name):
        base = name.find(args.prefix)
        offset = name.find(args.separater)
        PREFIX_LEN = len(args.prefix)
        index = name[base + PREFIX_LEN : offset]
        return int(index)

    def _extract_sync_and_async_ops(self, loss_value):
        gvs = self.optimizer.compute_gradients(loss_value)

        sync_ops = []
        async_ops = []

        for g, v in gvs:
            if g is None:
                continue
            if is_async_op(g):
                async_ops.append([g, v])
            else:
                sync_ops.append([g, v])

        return sync_ops, async_ops

    def _create_polling_op(self, DUMMY_PLACEHOLDER, op_index, shape, polling_op_name):
        polling_op = polling_data(
            DUMMY_PLACEHOLDER,
            op_index=op_index,
            s0=shape[0],
            s1=shape[1],
            s2=shape[2],
            s3=shape[3]
        )

        return polling_op

    def _set_applies_before_forwards(self, async_ops, DUMMY_PLACEHOLDER):
        if args.debug_print:
            print("\n\n======== ASYNC APPLY TO FORWARD OP ========")
            
        for g, v in async_ops:
            op_index = self._get_op_index(g.name)
            new_name = f"CONV_{op_index}"

            polling_op_name = f"POLLING_{new_name}"
            polling_op = self._create_polling_op(
                DUMMY_PLACEHOLDER, op_index, v.shape, polling_op_name
            )

            polling_gvs = []
            polling_gvs.append((polling_op, v))

            new_apply_name = f"NEXT_APPLY_{new_name}"
            apply_op = self.optimizer.apply_gradients(polling_gvs, name=new_apply_name)

            forward_op = self._find_forward_op(v, op_index)

            if args.debug_print:
                print_log(f"APPLY -> FORWARD_CONTROL_DEPENDENCY: {apply_op.name}    ->    {forward_op.op.name}")
            forward_op.op._add_control_input(apply_op)

            # for perfomance
            BEFORE_OP_INDEX = op_index - args.num_dependency
            if BEFORE_OP_INDEX >= 0:
                if args.debug_print:
                    print_log(f"FORWARD -> POLLING_CONTROL_DEPENDENCY: {before_forward.op.name}    ->    {polling_op.op.name}")

                before_forward = self._find_forward_op(v, BEFORE_OP_INDEX)
                polling_op.op._add_control_input(before_forward.op)

        if args.debug_print:
            print("====== END ASYNC APPLY TO FORWARD OP ======")
    
    def _reverse_k_schedule(self, graph):
        if args.debug_print:
            print("\n\n============ REVERSE OP for K ============")

        conv_weight_grads = {}
        for op in graph.get_operations():
            if is_send_recv_op(op):
                continue
            if is_async_op(op) is not True:
                continue
            if is_w_grad_op(op):
                op_index = self._get_op_index(op.name)
                conv_weight_grads[op_index] = op

        grads_data = conv_weight_grads
        size = len(grads_data) - 1
        for wop_index in grads_data:
            if wop_index == size:
                continue

            if args.debug_print:
                print_log(f"W_GRAD -> W_GRAD CONTROL_DEPENDENCY: \n    {cur_weight_grad.name}    ->    {next_weight_grad.name}")

            cur_weight_grad = grads_data[wop_index]
            next_weight_grad = grads_data[wop_index + 1]
            next_weight_grad._add_control_input(cur_weight_grad)

        if args.debug_print:
            print("=========== END REVERSE OP for K ==========")

    def schedule_ops(self, graph, X, loss_value, global_step):

        """
        g1, g2, g3, g4, g5
         |   |   |   |   |
         c   c   c   c   c
         |   |   |   |   |
         u   u   u   u   u
         =================
                                                        
                          P      P                              |                   P4     P5
                          |      |                              |                    |      |
                          u      u                              |                   u4     u5
                          |      |                              |                    |      |
        F1 -> F2 -> F3 -> F4 -> F5 ->  g1,  g2,   g3,  g4,  g5  |  F1 -> F2 -> F3 -> F4 -> F5
                                        |    |     |    |    |  |                                        c    c     c   c4   c5  |    
                                        |    |     |    |    |  |
                                        u1   u2   u3    Z    Z  |
        """

        sync_ops, async_ops = self._extract_sync_and_async_ops(loss_value)
        self._set_applies_before_forwards(async_ops, X)
        self._reverse_k_schedule(graph)

        # if args.debug_print:
        #     print_graph(graph)
        
        async_send_recv_ops = []
        for g, v in async_ops:
            async_send_recv_ops.append(comm_terminal(g))

        sync_apply = self.optimizer.apply_gradients(sync_ops, global_step=global_step)
        train_op = [sync_apply, async_send_recv_ops]

        return train_op

