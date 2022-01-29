import os
import sys
import math
import time

import numpy as np
import tensorflow as tf
import byteps.tensorflow as byteps

from OOO_backprop import get_args
from OOO_backprop import print_timestep
from OOO_backprop import print_log
from OOO_backprop.models import ResNet
from OOO_backprop.dp_schedule import OOO_ScheduleHelper

tf.compat.v1.disable_eager_execution()
args = None

def get_dataset(batch_size, dataset_str="MNIST"):
    if dataset_str == "MNIST":
        num_classes = 10

        dataset = tf.data.Dataset.from_tensor_slices(
            (tf.cast(mnist_images[..., tf.newaxis] / 255.0, tf.float32),
                    tf.cast(mnist_labels, tf.int64))
        )
        dataset = dataset.repeat().shuffle(10000).batch(batch_size)
    
    else: # dummy data
        num_classes = 1001

        x_data = np.array([np.ones((224, 224, 3), dtype=np.float32)])
        y_data = np.array([np.ones((1), dtype=np.int32)])
        y_data = tf.squeeze(tf.one_hot(y_data, depth=num_classes), 0)
        
        dataset = tf.data.Dataset.from_tensor_slices((x_data, y_data))
        dataset = dataset.repeat().batch(batch_size)    

    dataset = dataset.prefetch(batch_size)
    return dataset, num_classes

def train():
    global args

    # set dataset
    dataset, num_classes = get_dataset(args.batch_size, "Dummy")
    iterator = tf.compat.v1.data.make_initializable_iterator(dataset)

    # set model, loss, optimizer
    model = ResNet(depth=args.model_size, num_classes=num_classes)
    loss = tf.keras.losses.CategoricalCrossentropy()
    optimizer = tf.compat.v1.train.RMSPropOptimizer(0.001 * byteps.size())
    optimizer = byteps.DistributedOptimizer(optimizer)


    # construct a graph with calculating forward
    X, labels = iterator.get_next()
    probs = model(X, training=True)
    loss_value = loss(labels, probs)


    # get last training operations from dafault graph
    graph = tf.compat.v1.get_default_graph()
    global_step = tf.compat.v1.train.get_or_create_global_step(graph)

    # if args.debug_print == True:
    #     print("=============== data shape ==============")
    #     print_log(X.shape)
    #     print_log(labels.shape)
    #     print("=========================================")

    #     print("=============== print ops ===============")
    #     conv_ops_list = []
    #     async_count = 0
    #     for op in graph.get_operations():
    #         if "CONV_LAYER_" in op.name:
    #         # if "CONV_LAYER_" in op.name and len(op.name.split("/")) == 2 and "Conv2D" in op.name:
    #             conv_ops_list.append(op)
    #             if args.async_op in op.name:
    #                 async_count += 1
        
    #     print_log(f"num of CONV: {len(conv_ops_list)}, num of Async_CONV: {async_count}")

    #     for op in conv_ops_list:
    #         print_log(f"{op.name}", "B")
    #         print_log(f"{op.control_inputs}", "Y")
    #         print_log(f"{op.op_def}", "G")
    #     print("=========================================")

    schedule_helper = OOO_ScheduleHelper(optimizer)
    train_op = schedule_helper.schedule_ops(
        tf.compat.v1.get_default_graph(), X, loss_value, global_step
    )
    
    # set up config
    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(byteps.local_rank())
    hooks = [byteps.BroadcastGlobalVariablesHook(0)]

    # execute graph
    with tf.compat.v1.train.MonitoredTrainingSession(hooks=hooks, config=config) as sess:
        # initialize dataset
        sess.run(iterator.initializer)
        
        # training
        for run_count in range(args.num_training_step):
            with print_timestep(f"{run_count} iteration", average=False):
                _ = sess.run([train_op])

def main(_):
    train()

if __name__ == "__main__":
    args = get_args()
    byteps.init()

    tf.compat.v1.app.run(main)