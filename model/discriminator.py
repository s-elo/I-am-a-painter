import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras.layers import Conv2D, LeakyReLU, Input, ZeroPadding2D
from tensorflow.keras.initializers import TruncatedNormal
# from tensorflow_examples.models.pix2pix import pix2pix
from model.utils import instance_norm, down_sample

OUTPUT_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def get_discriminator():
    # g = pix2pix.discriminator(norm_type='instancenorm', target=False)
    # g = cnn_discriminator()
    g = normal_discriminator()
    return g


def normal_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    gamma_init = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.02)

    inp = Input(shape=[256, 256, 3], name='input_image')

    x = inp

    down1 = down_sample(64, 4, False)(x)  # (bs, 128, 128, 64)
    down2 = down_sample(128, 4)(down1)  # (bs, 64, 64, 128)
    down3 = down_sample(256, 4)(down2)  # (bs, 32, 32, 256)

    zero_pad1 = tf.keras.layers.ZeroPadding2D()(down3)  # (bs, 34, 34, 256)
    conv = tf.keras.layers.Conv2D(512, 4, strides=1,
                                  kernel_initializer=initializer,
                                  use_bias=False)(zero_pad1)  # (bs, 31, 31, 512)

    norm1 = tfa.layers.InstanceNormalization(
        gamma_initializer=gamma_init)(conv)

    leaky_relu = LeakyReLU()(norm1)

    zero_pad2 = ZeroPadding2D()(leaky_relu)  # (bs, 33, 33, 512)

    last = Conv2D(1, 4, strides=1,
                  kernel_initializer=initializer)(zero_pad2)  # (bs, 30, 30, 1)

    return tf.keras.Model(inputs=inp, outputs=last)


def cnn_discriminator():
    f = 4
    output_channels = 64

    cnn_input = tf.keras.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, OUTPUT_CHANNELS))

    cnn_output = Conv2D(filters=output_channels,
                        kernel_size=(f, f),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input)
    cnn_output = LeakyReLU(alpha=0.2)(cnn_output)

    cnn_output = Conv2D(filters=output_channels * 2,
                        kernel_size=(f, f),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input)
    cnn_output = instance_norm(cnn_output)
    cnn_output = LeakyReLU(alpha=0.2)(cnn_output)
    cnn_output = Conv2D(filters=output_channels * 4,
                        kernel_size=(f, f),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input)
    cnn_output = instance_norm(cnn_output)
    cnn_output = LeakyReLU(alpha=0.2)(cnn_output)
    cnn_output = Conv2D(filters=output_channels * 8,
                        kernel_size=(f, f),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input)
    cnn_output = instance_norm(cnn_output)
    cnn_output = LeakyReLU(alpha=0.2)(cnn_output)

    cnn_output = Conv2D(filters=1,
                        kernel_size=(f, f),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(cnn_input)

    return tf.keras.Model(inputs=cnn_input, outputs=cnn_output)
