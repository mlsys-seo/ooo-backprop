import tensorflow as tf
# import tensorflow.contrib.graph_editor as ge
# import graph_def_editor as gde

from OOO_backprop import get_args
from OOO_backprop import print_log
import byteps
args = get_args()

def encode_sched_map_into_name(layer_name, curr_layer_num=-1): 
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
    if "CONV_LAYER_" not in layer_name:
        return layer_name

    if curr_layer_num < args.reverse_first_k:
        print( "set async op", layer_name )
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
    '''Helper Object for adopting OOO schedule.
    '''
    def __init__(self, optimizer, num_worker):

        self.optimizer = optimizer
        # self.C_LIB = optimizer.get_push_lib()
        self.COMMUNICATION_CONV = 0

        global args

        self.prefix = args.prefix
        self.separater = args.separater
        self.async_op = args.async_op
        self.reverse_first_k = args.reverse_first_k
        self.num_dependency = args.num_dependency
        self.num_worker = num_worker

    def _is_async_op(self, op):
        if self.async_op in op.name:
            return True
        else:
            return False

    def _is_send_recv_op(self, op):
        if "Push_Pull" in op.name:
            return True
        else:
            return False

    def _is_w_grad_op(self, op):
        if "Conv2DBackpropFilter" in op.name:
            return True
        else:
            return False

    def _remove_w_grad_dependency(self, op):
        control_dependencies = op.control_inputs
        op._remove_all_control_inputs()

        # pop weight grad op
        control_dependencies = [ dep_op for dep_op in control_dependencies if not self._is_w_grad_op(dep_op)]
        op._add_control_inputs(control_dependencies)

    def _find_forward_op(self, v, key):
        forward_op = forward_ops_table[key]
        return forward_op

    def _get_op_index(self, name):
        base = name.find(self.prefix)
        offset = name.find(self.separater)
        PREFIX_LEN = len(self.prefix)
        index = name[base + PREFIX_LEN : offset]
        return int(index)

    def _extract_sync_and_async_ops(self, loss_value):
        print("_extract_sync_and_async_ops ################################")
        gvs = self.optimizer.compute_gradients(loss_value)

        sync_ops = []
        async_ops = []

        for g, v in gvs:
            if g is None:
                continue
            if self._is_async_op(g):
                print("\t ASYNC G ", g.name)
                async_ops.append([g, v])
            else:
                print("\t SYNC  G ", g.name)
                sync_ops.append([g, v])

        return sync_ops, async_ops

    def _create_polling_op(self, DUMMY_PLACEHOLDER, op_index, shape, polling_op_name):
        #polling_op = byteps.tensorflow.wait_data(
        polling_op = byteps.tensorflow.polling_data(
            DUMMY_PLACEHOLDER,
            op_index=op_index,
            s0=shape[0],
            s1=shape[1],
            s2=shape[2],
            s3=shape[3],
        )

        return polling_op

    def _set_applies_before_forwards(self, async_ops, DUMMY_PLACEHOLDER):
        polling_ops = []
        for g, v in async_ops:
            op_index = self._get_op_index(g.name)
            new_name = f"CONV_{op_index}"

            polling_op_name = f"POLLING_{new_name}"
            print( g.op )

            dummy_X = tf.stop_gradient(
                tf.Variable(
                    tf.zeros( g.shape ), name="input_data", dtype="float"
                )
            )

            polling_op = self._create_polling_op(
                dummy_X, op_index, v.shape, polling_op_name
            )
            
            #polling_op = self._create_polling_op(
            #    DUMMY_PLACEHOLDER, op_index, v.shape, polling_op_name
            #)
            polling_op = tf.math.divide( polling_op, self.num_worker )

            polling_gvs = []
            polling_gvs.append((polling_op, v))

            new_apply_name = f"NEXT_APPLY_{new_name}"
            apply_op = self.optimizer.apply_gradients(polling_gvs, name=new_apply_name)

            forward_op = self._find_forward_op(v, op_index)

            # ge.add_control_inputs(forward_op.op, [apply_op])
            forward_op.op._add_control_input(apply_op)

            #for perfomance
            BEFORE_OP_INDEX = op_index - self.num_dependency
            if BEFORE_OP_INDEX >= 0:
                before_forward = self._find_forward_op(v, BEFORE_OP_INDEX)
                # ge.add_control_inputs(polling_op.op, [before_forward.op])
                polling_op.op._add_control_input(before_forward.op)

            polling_ops.append(polling_op)

        return polling_ops

    def _reverse_k_schedule(self, graph):
        conv_weight_grads = {}
        for op in graph.get_operations():
            if self._is_send_recv_op(op):
                continue
            if self._is_async_op(op) is not True:
                continue
            if self._is_w_grad_op(op):
                op_index = self._get_op_index(op.name)
                conv_weight_grads[op_index] = op

            #### remove control dependency
            # if "Conv2D_grad/tuple/group_deps" in op.name:
            #     self._remove_w_grad_dependency(op)


        grads_data = conv_weight_grads
        size = len(grads_data) - 1
        for wop_index in grads_data:
            if wop_index == size:
                continue

            cur_weight_grad = grads_data[wop_index] # layer 1
            next_weight_grad = grads_data[wop_index + 1] # layer 2
            # ge.add_control_inputs(next_weight_grad, [cur_weight_grad])
            next_weight_grad._add_control_input(cur_weight_grad)

    def schedule_ops(self, graph, X, loss_value, global_step):

        """
        gvs = self.optimizer.compute_gradientst(cost)
        train_op = self.optimizer.apply_gradients(gvs)

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

        sync_apply = ( u1, u2, u3 )
        train_op = ( [u1,u2,u3],  [c4, c5] )

        5->4->3->2->1 |SYNC| origin

                      |next forward
        1->2->3 |SYNC|->4->5 - our


        """


        sync_ops, async_ops = self._extract_sync_and_async_ops(loss_value)
        
        print_log(f"num of CONV: {len(sync_ops)}, num of Async_CONV: {len(async_ops)}")

        #for op in conv_ops_list:
        #    print_log(f"OP NAME: {op.name}", "B")
        #    print_log(f"INPUTS: {[ target.name for target in op.inputs ]}", "Y")
        #    print_log(f"OUTPUTS: {[ target.name for target in op.outputs ]}", "G")
        #    print_log(f"CONTROL_INPUTS: {[ target.name for target in op.control_inputs ]}", "C")
        #    print(" ")
        #    # print_log(f"{op.op_def}", "G")
        print("=========================================")

        polling_ops = self._set_applies_before_forwards(async_ops, X)
        self._reverse_k_schedule(graph)

        # print("======= print ops in dp_schedule ========")
        # conv_ops_list = []
        # async_count = 0
        # for op in graph.get_operations():
        #     # if "CONV_LAYER_" in op.name:
        #     if "gradients" in op.name:
        #     # if "CONV_LAYER_" in op.name and "Conv2D" in op.name:
        #         conv_ops_list.append(op)
        #         if args.async_op in op.name:
        #             async_count += 1
        
        # print_log(f"num of CONV: {len(conv_ops_list)}, num of Async_CONV: {async_count}")

        # for op in conv_ops_list:
        #     print_log(f"{op.name}", "B")
        #     print_log(f"{op.control_inputs}", "Y")
        #     # print_log(f"{op.op_def}", "G")
        # print("=========================================")

        # async_send_recv_helper = tf.load_op_library(
        #     "./async_send_recv_helper.so"
        # )
        async_send_recv_ops = []
        for g, v in async_ops:
            # async_send_recv_ops.append(async_send_recv_helper.remove_zero_copy(g))
            async_send_recv_ops.append(g)

        sync_apply = self.optimizer.apply_gradients(sync_ops, global_step=global_step)
        train_op = [sync_apply, async_send_recv_ops]

        return train_op, sync_ops, async_ops, polling_ops

