import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow_addons as tfa


def res_block(res_input, output_channels):
    res_output = tf.pad(res_input, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    res_output = Conv2D(filters=output_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)
    res_output = tf.keras.activations.relu(res_output)

    res_output = tf.pad(
        res_output, [[0, 0], [1, 1], [1, 1], [0, 0]], "REFLECT")
    res_output = Conv2D(filters=output_channels,
                        kernel_size=(3, 3),
                        strides=(1, 1),
                        padding='valid',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)

    # input + F(input)
    return tf.keras.activations.relu(res_output + res_input)


def down_sample(filters, size, apply_instancenorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    result = tf.keras.Sequential()
    result.add(Conv2D(filters, size, strides=2, padding='same',
                      kernel_initializer=initializer, use_bias=False))

    if apply_instancenorm:
        result.add(tfa.layers.InstanceNormalization(
            gamma_initializer=gamma_init))

    result.add(LeakyReLU())

    return result


def up_sample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    layer = tf.keras.Sequential()
    layer.add(tf.keras.layers.Conv2DTranspose(filters, size, strides=2,
              padding='same', kernel_initializer=initializer, use_bias=False))
    layer.add(tfa.layers.InstanceNormalization(gamma_initializer=gamma_init))

    if apply_dropout:
        layer.add(tf.keras.layers.Dropout(0.5))

    layer.add(tf.keras.layers.ReLU())

    return layer


def instance_norm(x):
    return tfa.layers.InstanceNormalization()(x)
