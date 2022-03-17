import tensorflow as tf
import byteps.tensorflow as byteps
from tensorflow import keras

from tensorflow.python.training import training_ops
import numpy as np
import time
from OOO_backprop import dp_schedule
from OOO_backprop import get_args
from OOO_backprop import print_timestep
from OOO_backprop import print_log
from OOO_backprop.dp_schedule import OOO_ScheduleHelper

import random

import os
import sys
import math



SEED = 1
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

tf.compat.v1.disable_eager_execution()
_curr_layer_num = 0
BATCH_SIZE = 1
filter_size = 1
LR = 0.0001
img_rows, img_cols, img_channels = 6, 6, 3
num_classes = 128

def conv2d(inputs, filters, kernel_size, stride=1, name=""):
    if stride > 1:
        inputs = pad(inputs, kernel_size)
        pad_scheme = "VALID"
    else:
        pad_scheme = "SAME"

    conv_op = tf.compat.v1.layers.conv2d(
        inputs=inputs,
        filters=filters,
        kernel_size=kernel_size,
        strides=stride,
        padding=pad_scheme,
        use_bias=False,
        kernel_initializer=tf.keras.initializers.Constant(value=0.01),
        data_format="channels_first",
        name=name,
    )
    return conv_op

def _conv_layer(x, filters, kernel_size, stride=1):
    global _curr_layer_num
    layer_name = f"CONV_LAYER_"
    conv = conv2d(
        x,
        filters,
        kernel_size,
        stride,
        name=dp_schedule.encode_sched_map_into_name(
            layer_name, _curr_layer_num
        ),
    )

    dp_schedule.collect_conv_op(conv, _curr_layer_num)
    _curr_layer_num += 1
    return conv

def train():
    x_batch = np.random.rand(BATCH_SIZE, img_channels, img_rows, img_cols)
    y_batch = np.random.rand(BATCH_SIZE, num_classes)
    X = tf.compat.v1.placeholder(tf.compat.v1.float32, shape=(BATCH_SIZE, img_channels, img_rows, img_cols))
    Y = tf.compat.v1.placeholder(tf.compat.v1.int64, [None, num_classes] )

    layer1 = _conv_layer(X, filter_size, kernel_size=[3,3], stride=1)
    layer1 = tf.compat.v1.nn.relu(layer1)
    layer2 = _conv_layer(layer1, filter_size, kernel_size=[3,3], stride=1)
    layer2 = tf.compat.v1.nn.relu(layer2)
    layer3 = _conv_layer(layer2, filter_size, kernel_size=[3,3], stride=1)

    flatten_size = layer3.shape[0] * layer3.shape[1] *  layer3.shape[2] * layer3.shape[3]
    layer3_flatten = tf.compat.v1.reshape(layer3, [-1, flatten_size])
    w1 = tf.compat.v1.Variable( np.random.rand(flatten_size, num_classes), dtype="float32", name = "w1" ) 
    logit = tf.compat.v1.matmul( layer3_flatten, w1 )
    cost = tf.reduce_mean(
        input_tensor = tf.compat.v1.losses.softmax_cross_entropy( Y, logit )
    )

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LR)
    optimizer = byteps.DistributedOptimizer(optimizer)

    graph = tf.compat.v1.get_default_graph()
    global_step = tf.compat.v1.train.get_or_create_global_step(graph)

    schedule_helper = OOO_ScheduleHelper(optimizer, byteps.size())
    train_op, sync_ops, async_ops, polling_ops = schedule_helper.schedule_ops(
        tf.compat.v1.get_default_graph(), X, cost, global_step
    )

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(byteps.local_rank())
    hooks = [byteps.BroadcastGlobalVariablesHook(0)]

    sess = tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks, config=config)
    sess.graph._unsafe_unfinalize()
    v = sess.run(tf.compat.v1.global_variables_initializer())

    TRAIN_STEP = args.num_training_step
    print( "TRAIN_STEP : ", TRAIN_STEP )
    file_name = "./logit_base"
    if( args.reverse_first_k ):
        layer1_com_op = polling_ops[0]
        print( layer1_com_op )
        TRAIN_STEP += 1
        ile_name = "./logit_ooo"
    else:
        layer1_com_op = sync_ops[0][0]
        print( layer1_com_op )

    for step in range(TRAIN_STEP):
      _, w_grad_val = sess.run( [train_op, layer1_com_op], feed_dict={X:x_batch, Y:y_batch} )

    print("STEP : ", TRAIN_STEP)
    result = w_grad_val * LR
    print( result )


def main(_):
    train()

if __name__ == "__main__":
    args = get_args()
    byteps.init()
    tf.compat.v1.app.run(main)

