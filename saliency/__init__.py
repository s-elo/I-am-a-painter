import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
import numpy as np


def get_saliency_map(images, labels):
    model = VGG16(weights='imagenet')

    expected_outputs = to_categorical(labels, num_classes=1000)

    with tf.GradientTape() as tape:
        images = tf.cast(images, tf.float32)
        tape.watch(images)

        preds = model(images, training=False)

        results = decode_predictions(preds.numpy(), top=1)

        losses = tf.keras.losses.categorical_crossentropy(
            expected_outputs, preds)

    grads = tape.gradient(losses, images)
    # supress the channel dimension
    grayscale_tensors = tf.reduce_sum(tf.abs(grads), axis=-1)
    # normalizaion
    normalized_tensors = tf.cast(255 * (grayscale_tensors - tf.reduce_min(grayscale_tensors)) / (
        tf.reduce_max(grayscale_tensors) - tf.reduce_min(grayscale_tensors)), tf.uint8)
    normalized_tensors = tf.squeeze(normalized_tensors)

    return normalized_tensors, results
