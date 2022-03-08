import sys


def check_right_schedule(layer_num, colocated_layer_num):
    if layer_num < colocated_layer_num:
        print("ERROR: Wrong schedule.")
        sys.exit(1)


def encode_sched_map_into_name(layer_name, curr_layer_num=-1, w_grad_sched_map=None):
    assert curr_layer_num > -1
    assert w_grad_sched_map != None

    current_schedule = w_grad_sched_map[curr_layer_num]
    colocation_method = current_schedule[0]
    if colocation_method == "NO":
        encoded_layer_name = layer_name
    else:
        colocated_layer_num = current_schedule[1]
        check_right_schedule(curr_layer_num, colocated_layer_num) 

        colocated_layer_name = colocation_method + str(colocated_layer_num)
        if curr_layer_num == colocated_layer_num:
            encoded_layer_name = "_".join(
                [
                    layer_name,
                    "SUB_STREAM",
                ]
            )
        else:
            encoded_layer_name = "_".join(
                [
                    layer_name,
                    colocated_layer_name,
                    "SUB_STREAM",
                ]
            )

    return encoded_layer_name
