import tensorflow as tf
from tensorflow.keras.layers import Conv2D, LeakyReLU
from tensorflow.keras.initializers import TruncatedNormal
import tensorflow_addons as tfa
import PIL
import numpy as np
import cv2


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


def load_models(load_path='./saved_models', model='all'):
    if (model != 'all'):
        return tf.keras.models.load_model(f'{load_path}/{model}', compile=False)
    else:
        m_gen = tf.keras.models.load_model(
            f'{load_path}/m_gen', compile=False)
        p_gen = tf.keras.models.load_model(
            f'{load_path}/p_gen', compile=False)
        m_disc = tf.keras.models.load_model(
            f'{load_path}/m_disc', compile=False)
        p_disc = tf.keras.models.load_model(
            f'{load_path}/p_disc', compile=False)

        return m_gen, p_gen, m_disc, p_disc


def get_monet_style(imgs, monet_generator, output_shape):
    ret = []

    for img in imgs:
        # resize and reshape for the generator
        img = cv2.resize(img, (256, 256),
                         interpolation=cv2.INTER_CUBIC)
        # the img has been processed by /255 before
        img = np.array(img) * 255
        img = decode_image(img, (256, 256))

        gen_img = monet_generator(img, training=False)[0].numpy()
        gen_img = (gen_img * 127.5 + 127.5).astype(np.uint8)

        # reshape back
        gen_img = cv2.resize(gen_img, output_shape,
                             interpolation=cv2.INTER_CUBIC)

        ret.append(gen_img)

    return np.array(ret)


def decode_image(image, IMAGE_SIZE):
    # image = PIL.Image.fromarray(image)
    # image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [1, *IMAGE_SIZE, 3])
    return image
