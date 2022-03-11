import os
import sys
import time
import numpy as np
import tensorflow as tf
from tensorflow.python.training import training_ops

import encode
import mobilenet_v3_schedule_map

tf.compat.v1.disable_eager_execution()

BATCH = int(sys.argv[1])
ALPHA = float(sys.argv[2])

layers = [
    [16, 16, 3, 1, "RE", False, 16],
    [16, 24, 3, 2, "RE", False, 64],
    [24, 24, 3, 1, "RE", False, 72],
    [24, 40, 5, 2, "RE", True, 72],
    [40, 40, 5, 1, "RE", True, 120],

    [40, 40, 5, 1, "RE", True, 120],
    [40, 80, 3, 2, "HS", False, 240],
    [80, 80, 3, 1, "HS", False, 200],
    [80, 80, 3, 1, "HS", False, 184],
    [80, 80, 3, 1, "HS", False, 184],

    [80, 112, 3, 1, "HS", True, 480],
    [112, 112, 3, 1, "HS", True, 672],
    [112, 160, 5, 1, "HS", True, 672],
    [160, 160, 5, 2, "HS", True, 672],
    [160, 160, 5, 1, "HS", True, 960],
]


def make_divisible(v, divisor=8, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def relu6(x, name='relu6'):
    return tf.nn.relu6(x, name)


def hard_swish(x, name='hard_swish'):
    with tf.compat.v1.variable_scope(name):
        h_swish = x * tf.nn.relu6(x + 3) / 6
    return h_swish


def hard_sigmoid(x, name='hard_sigmoid'):
    with tf.compat.v1.variable_scope(name):
        h_sigmoid = tf.nn.relu6(x + 3) / 6
    return h_sigmoid


def global_avg(x, pool_size, strides, padding='valid'):
    return tf.compat.v1.layers.average_pooling2d(
        inputs=x,
        pool_size=pool_size,
        strides=strides,
        padding=padding,
        data_format='channels_last',
        name='global_avg',
    )


class MobileNetV3:
    def __init__(self, x, num_classes, layers, multiplier=1.0, reduction_ratio=4):
        self.num_layers = 0
        self.multiplier = multiplier
        self.reduction_ratio = reduction_ratio
        self.num_classes = num_classes
        self.layers = layers
        self.model = self._make_model(x)

    def _conv2d_layer(self, x, filters, kernel, use_bias=False, strides=1, padding="SAME"):
        layer_name = '_'.join(["CONV", str(self.num_layers)])

        x = tf.compat.v1.layers.conv2d(
            inputs=x,
            filters=filters,
            kernel_size=kernel,
            strides=strides,
            kernel_initializer=tf.compat.v1.glorot_uniform_initializer(),
            padding=padding,
            kernel_regularizer=tf.keras.regularizers.l2(l=0.5 * 5e-4),
            use_bias=use_bias,
            name=encode.encode_sched_map_into_name(
                layer_name,
                self.num_layers,
                mobilenet_v3_schedule_map.weight_gradient_schedule_map,
            ),
        )

        return x

    def _dwise_conv(self, x, k_h=3, k_w=3, depth_multiplier=1, strides=(1, 1), padding='SAME'):
        layer_name = '_'.join(["DWCONV", str(self.num_layers)])

        kernel_size = (k_w, k_h)
        in_channel = x.get_shape().as_list()[-1]
        filters = int(in_channel * depth_multiplier)
        x = tf.compat.v1.layers.separable_conv2d(
            x, filters, kernel_size,
            strides=strides, padding=padding,
            data_format='channels_last', dilation_rate=(1, 1),
            depth_multiplier=depth_multiplier, activation=None,
            use_bias=False,
            name=encode.encode_sched_map_into_name(
                layer_name,
                self.num_layers,
                mobilenet_v3_schedule_map.weight_gradient_schedule_map,
            ),
        )

        return x

    def _fully_connected_layer(self, x, units, name="fc", activation=None, use_bias=True):
        layer_name = '_'.join([name, str(self.num_layers)])
        x = tf.compat.v1.layers.dense(x, units=units, activation=activation, use_bias=use_bias, name=layer_name)

        return x

    def _squeeze_excitation_layer(self, x, out_dim, ratio):
        with tf.compat.v1.variable_scope('se_block'):
            _x = global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)
            _x = self._fully_connected_layer(_x, units=out_dim / ratio, name='se_block_excitation1')
            _x = relu6(_x)
            _x = self._fully_connected_layer(_x, units=out_dim, name='se_block_excitation2')
            _x = hard_sigmoid(_x)

            _x = tf.reshape(_x, [-1, 1, 1, out_dim])
            scale = x * _x

            return scale

    def _batch_normalization_layer(self, x, momentum=0.997, epsilon=1e-3):
        layer_name = "BN_" + str(self.num_layers)
        self.num_layers += 1
        return tf.compat.v1.layers.batch_normalization(
            inputs=x,
            momentum=momentum,
            epsilon=epsilon,
            scale=True,
            center=True,
            name=layer_name,
        )

    def _conv_1x1_bn(self, x, filters):
        x = self._conv2d_layer(x, filters, 1, use_bias=False, strides=1)
        x = self._batch_normalization_layer(x, momentum=0.997, epsilon=1e-3)

        return x

    def _conv_bn_relu(self, x, filters, kernel, strides=1, activation=relu6):
        x = self._conv2d_layer(x, filters, kernel, use_bias=False, strides=strides)
        x = self._batch_normalization_layer(x, momentum=0.997, epsilon=1e-3)
        x = activation(x)

        return x

    def _activation(self, x, activation="RE"):
        if activation == "HS":
            return hard_swish(x)
        elif activation == "RE":
            return relu6(x)
        else:
            raise NotImplementedError

    def mobilenet_v3_block(self, x, layer_info, idx):
        (in_channels, out_channels, kernel, stride, activation, se, exp_size) = layer_info
        in_channels = make_divisible(in_channels * self.multiplier)
        out_channels = make_divisible(out_channels * self.multiplier)
        bottleneck_dim = make_divisible(exp_size * self.multiplier)

        with tf.compat.v1.variable_scope(f"bneck{idx}"):
            _x = self._conv_1x1_bn(x, bottleneck_dim)
            _x = self._activation(_x, activation)
            _x = self._dwise_conv(_x, k_w=kernel, k_h=kernel, strides=[stride, stride])
            _x = self._batch_normalization_layer(_x, momentum=0.997, epsilon=1e-3)
            _x = self._activation(_x, activation)

            if se:
                channel = _x.get_shape().as_list()[-1]
                _x = self._squeeze_excitation_layer(_x, out_dim=channel, ratio=self.reduction_ratio)

            _x = self._conv_1x1_bn(_x, out_channels)

            if in_channels == out_channels and stride == 1:
                _x += x
                _x = tf.identity(_x, name='block_output')

        return _x

    def _make_model(self, x):
        input_size = x.get_shape().as_list()[1:-1]
        assert ((input_size[0] % 32 == 0) and (input_size[1] % 32 == 0))

        with tf.compat.v1.variable_scope('Init'):
            init_conv_out = make_divisible(16 * self.multiplier)
            x = self._conv_bn_relu(x, filters=init_conv_out, kernel=3, strides=2, activation=hard_swish)

        with tf.compat.v1.variable_scope("MobilenetV3_large"):
            for idx, layer_info in enumerate(self.layers):
                x = self.mobilenet_v3_block(x, layer_info, idx)

            conv1_out = make_divisible(960 * self.multiplier)
            x = self._conv_bn_relu(x, filters=conv1_out, kernel=1, strides=1, activation=hard_swish)

            x = global_avg(x, pool_size=x.get_shape()[1:-1], strides=1)

        with tf.compat.v1.variable_scope('Logits_out'):
            conv2_out = make_divisible(1280 * self.multiplier)
            x = self._conv2d_layer(x, filters=conv2_out, kernel=1, use_bias=True, strides=1)
            self.num_layers += 1
            x = hard_swish(x)
            x = self._conv2d_layer(x, filters=self.num_classes, kernel=1, use_bias=True, strides=1)
            self.num_layers += 1
            x = tf.compat.v1.layers.flatten(x, name='output')

        return x


if __name__ == "__main__":
    num_classes = 1000

    dummy_X = tf.stop_gradient(
        tf.Variable(
            tf.random.normal([BATCH, 224, 224, 3]), name="input_data", dtype="float"
        )
    )
    dummy_Y = tf.stop_gradient(
        tf.Variable(
            tf.random.normal([BATCH, num_classes]), name="lable", dtype="float"
        )
    )

    logits = MobileNetV3(dummy_X, num_classes, layers, multiplier=ALPHA, reduction_ratio=4).model
    optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.1)

    cost = tf.reduce_mean(
        input_tensor = tf.compat.v1.losses.softmax_cross_entropy( dummy_Y, logits )
    )

    gvs = optimizer.compute_gradients(cost, tf.compat.v1.trainable_variables())
    train_op = optimizer.apply_gradients(gvs)

    cuda_graph_run_op = training_ops.prepare_cuda_graph_capture()

    config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    with tf.compat.v1.Session(config=config) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())

        ################################################
        # Running MobileNet v3 with Tensorflow XLA  
        ################################################
        warmup_step = 3
        tf_graph_step = 50 + warmup_step
        oneiter_times = []
        print("========== XLA Training Start ==========")
        for step in range(tf_graph_step):
            start_time = time.time()
            sess.run([train_op])
            end_time = time.time()

            oneiter_time = end_time - start_time
            oneiter_times.append(oneiter_time)
        print("=========== XLA Training End ===========")

        # The execution time of the first few iteration is high because of TF graph initialization and GPU initialization overhead.
        # So it disturbs to measure the average execution time.
        oneiter_times.sort()
        for _ in range(3):
            oneiter_times.pop()

        avg_time = sum(oneiter_times) / len(oneiter_times)
        print("XLA Training Throughput : {:.2f} (images/sec)".format(BATCH / avg_time))

