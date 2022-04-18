import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Conv2DTranspose, Input, Conv2DTranspose
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow_examples.models.pix2pix import pix2pix
from model.utils import res_block, instance_norm, up_sample, down_sample

OUTPUT_CHANNELS = 3
IMAGE_WIDTH = 256
IMAGE_HEIGHT = 256


def get_generator():
    # g = pix2pix.unet_generator(OUTPUT_CHANNELS, norm_type='instancenorm')
    # g = res_generator()
    g = normal_generator()
    return g


def normal_generator():
    inputs = Input(shape=[256, 256, 3])
    down_stack = [
        down_sample(64, 4, apply_instancenorm=False),  # (size, 128, 128, 64)
        down_sample(128, 4),                         # (size, 64, 64, 128)
        down_sample(256, 4),                         # (size, 32, 32, 256)
        down_sample(512, 4),                         # (size, 16, 16, 512)
        down_sample(512, 4),                         # (size, 8, 8, 512)
        down_sample(512, 4),                         # (size, 4, 4, 512)
        down_sample(512, 4),                         # (size, 2, 2, 512)
        down_sample(512, 4),                         # (size, 1, 1, 512)
    ]

    up_stack = [
        up_sample(512, 4, apply_dropout=True),       # (size, 2, 2, 1024)
        up_sample(512, 4, apply_dropout=True),       # (size, 4, 4, 1024)
        up_sample(512, 4, apply_dropout=True),       # (size, 8, 8, 1024)
        up_sample(512, 4),                           # (size, 16, 16, 1024)
        up_sample(256, 4),                           # (size, 32, 32, 512)
        up_sample(128, 4),                           # (size, 64, 64, 256)
        up_sample(64, 4),                            # (size, 128, 128, 128)
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(3, 4, strides=2, padding='same',
                           kernel_initializer=initializer, activation='tanh')
    # (size, 256, 256, 3)

    x = inputs

    # Downsampling through the model
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def res_generator(res_block_num=9):
    f = 7
    ks = 3
    output_channels = 32

    res_input = tf.keras.Input(
        shape=(IMAGE_WIDTH, IMAGE_HEIGHT, OUTPUT_CHANNELS))
    # down sampling
    pad_input = tf.pad(res_input, [[0, 0], [ks, ks], [
                       ks, ks], [0, 0]], "REFLECT")
    res_output = Conv2D(filters=output_channels,
                        kernel_size=(f, f),
                        strides=(1, 1),
                        padding='valid',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(pad_input)
    res_output = instance_norm(res_output)
    res_output = tf.keras.activations.relu(res_output)
    res_output = Conv2D(filters=output_channels * 2,
                        kernel_size=(ks, ks),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)
    res_output = tf.keras.activations.relu(res_output)
    res_output = Conv2D(filters=output_channels * 4,
                        kernel_size=(ks, ks),
                        strides=(2, 2),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)
    res_output = tf.keras.activations.relu(res_output)

    # go through the res blocks
    for _ in range(0, res_block_num):
        res_output = res_block(res_output, output_channels * 4)

    # up sampling
    res_output = Conv2DTranspose(filters=output_channels * 2,
                                 kernel_size=(ks, ks),
                                 strides=(2, 2),
                                 padding='same',
                                 kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)
    res_output = tf.keras.activations.relu(res_output)
    res_output = Conv2DTranspose(filters=output_channels,
                                 kernel_size=(ks, ks),
                                 strides=(2, 2),
                                 padding='same',
                                 kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)
    res_output = tf.keras.activations.relu(res_output)
    res_output = Conv2D(filters=OUTPUT_CHANNELS,
                        kernel_size=(f, f),
                        strides=(1, 1),
                        padding='same',
                        kernel_initializer=TruncatedNormal(stddev=0.02))(res_output)
    res_output = instance_norm(res_output)

    res_output = tf.keras.activations.tanh(res_output)

    return tf.keras.Model(inputs=res_input, outputs=res_output)
