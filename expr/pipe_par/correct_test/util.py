

GRADIENTS = "gradients"
W_GRAD = "WEIGHT_GRAD"
O_GRAD = "OUT_GRAD"

def schedule_ooo_backpropagation(graph):
    fc2_w_grad = None
    fc2_o_grad = None
    
    block2_w_grads = []
    block2_o_grads = []
    
    #find w,o grad
    for op in graph.get_operations():
        if( GRADIENTS not in op.name ):
            continue

        if( W_GRAD in op.name and "BLOCK_2" in op.name ):
            block2_w_grads.append(op)
    
        if( O_GRAD in op.name and "BLOCK_2" in op.name ):
            block2_o_grads.append(op)
    
    
    for w_grad in block2_w_grads:
        target_o_grad = block2_o_grads[-1]
        print(target_o_grad.name, "->", w_grad.name)
        w_grad._add_control_input( target_o_grad )
