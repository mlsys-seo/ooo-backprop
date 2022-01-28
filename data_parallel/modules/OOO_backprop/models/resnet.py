import tensorflow as tf

from OOO_backprop.schedulers import dp_schedule

def pad(inputs, kernel_size):
    pad_total = kernel_size - 1
    pad_begin = pad_total // 2
    pad_end = pad_total - pad_begin

    padded_inputs = tf.pad(
        inputs, paddings=[[0, 0], [0, 0], [pad_begin, pad_end], [pad_begin, pad_end]]
    )
    return padded_inputs


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
        kernel_initializer=tf.keras.initializers.VarianceScaling(
            scale=1.0, mode='fan_in', distribution='truncated_normal'
        ),
        data_format="channels_first",
        name=name,
    )
    return conv_op


def batch_norm(inputs, training, name=""):
    batch_layer = tf.compat.v1.layers.batch_normalization(
        inputs,
        axis=1,
        momentum=0.997,
        epsilon=1e-5,
        training=training,
        fused=True,
        name=name,
    )
    return batch_layer


def max_pooling(inputs, pool_size=[3, 3], stride=2, padding="SAME"):
    return tf.compat.v1.layers.max_pooling2d(
        inputs,
        pool_size=pool_size,
        strides=stride,
        padding=padding,
        data_format="channels_first",
    )


class ResNet(object):
    def __init__(self, depth, num_classes):
        assert (
            depth == 18
            or depth == 34
            or depth == 50
            or depth == 101
            or depth == 152
            or depth == 200
        )

        self._curr_layer_num = 0

        self._num_classes = num_classes
        self._num_filters = 64

        self._stride_per_block = [1, 2, 2, 2]

        if depth == 18:
            self._num_layers_per_block = [2, 2, 2, 2]
        elif depth == 34:
            self._num_layers_per_block = [3, 4, 6, 3]
        elif depth == 50:
            self._num_layers_per_block = [3, 4, 6, 3]
        elif depth == 101:
            self._num_layers_per_block = [3, 4, 23, 3]
        elif depth == 152:
            self._num_layers_per_block = [3, 8, 36, 3]
        elif depth == 200:
            self._num_layers_per_block = [3, 24, 36, 3]

        if depth <= 34:
            self._use_bottleneck = False
        else:
            self._use_bottleneck = True

    def _conv_layer(self, x, filters, kernel_size, stride=1):
        layer_name = f"CONV_LAYER_"
        conv = conv2d(
            x,
            filters,
            kernel_size,
            stride,
            name=dp_schedule.encode_sched_map_into_name(
                layer_name, self._curr_layer_num
            ),
        )
        dp_schedule.collect_conv_op(conv, self._curr_layer_num)
        self._curr_layer_num += 1
        return conv

    def _batch_norm_layer(self, x, training):
        layer_name = f"BATCH_LAYER_"
        bn = batch_norm(
            x,
            training,
            name=dp_schedule.encode_sched_map_into_name(
                layer_name, self._curr_layer_num
            ),
        )
        return bn

    def _bottleneck_block(self, x, filters, stride, projection_shortcut, training):
        shortcut = x

        if projection_shortcut is not None:
            shortcut = projection_shortcut(shortcut)
            shortcut = batch_norm(shortcut, training)

        x = self._conv_layer(x, filters, kernel_size=1, stride=1)
        x = self._batch_norm_layer(x, training)
        x = tf.nn.relu(x)

        x = self._conv_layer(x, filters, kernel_size=3, stride=stride)
        x = self._batch_norm_layer(x, training)
        x = tf.nn.relu(x)

        x = self._conv_layer(x, filters * 4, kernel_size=1, stride=1)
        x = self._batch_norm_layer(x, training)
        x += shortcut
        x = tf.nn.relu(x)

        return x

    def _block_layer(
        self, x, filters, block_stride, num_blocks, use_bottleneck, training
    ):
        block_fn = self._bottleneck_block if use_bottleneck else self._basic_block

        def projection_shortcut(x):
            if use_bottleneck:
                return self._conv_layer(x, filters * 4, 1, block_stride)
            else:
                return self._conv_layer(x, filters, 1, block_stride)

        x = block_fn(x, filters, block_stride, projection_shortcut, training)
        for _ in range(1, num_blocks):
            x = block_fn(x, filters, 1, None, training)

        return x

    def __call__(self, x, training=False):
        x = tf.transpose(a=x, perm=[0, 3, 1, 2])

        with tf.name_scope("initial_conv") as scope:
            x = self._conv_layer(x, 64, 7, 2)
            x = self._batch_norm_layer(x, training)
            x = tf.nn.relu(x)
            x = max_pooling(x, 3, 2)

        for i, num_blocks in enumerate(self._num_layers_per_block):
            num_filters = self._num_filters * (2 ** i)
            x = self._block_layer(
                x,
                num_filters,
                self._stride_per_block[i],
                num_blocks,
                self._use_bottleneck,
                training,
            )

        with tf.name_scope("flatten") as scope:
            x = tf.reduce_mean(input_tensor=x, axis=[2, 3], keepdims=True)
            x = tf.squeeze(x, [2, 3])

        with tf.name_scope("fc") as scope:
            x = tf.compat.v1.layers.dense(inputs=x, units=self._num_classes)

        return x