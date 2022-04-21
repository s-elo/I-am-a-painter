import tensorflow as tf
from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
# from tensorflow.keras.applications.vgg19 import preprocess_input, decode_predictions, VGG19
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions, InceptionV3
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input, decode_predictions
# from tensorflow.keras.applications.resnet50 import preprocess_input, decode_predictions, ResNet50
import numpy as np
from saliency.graph_cut import graph_cut


def get_saliency_map(images, labels):
    model = InceptionV3(weights='imagenet')

    expected_outputs = to_categorical(labels, num_classes=1000)

    smaps = []

    with tf.GradientTape() as tape:
        images = tf.cast(images, tf.float32)

        tape.watch(images)

        preds = model(images, training=False)

        results = decode_predictions(preds.numpy(), top=1)

        losses = tf.keras.losses.categorical_crossentropy(
            expected_outputs, preds)

    grads = tape.gradient(losses, images)

    # # take maximum across channels
    # gradients = tf.reduce_max(tf.abs(grads), axis=-1)

    # # normalization to 0-255
    # gradients = (gradients - np.min(gradients)) / (np.max(gradients) - np.min(gradients))
    # return gradients.numpy() * 255, results
    for grad in grads:
        # take maximum across channels
        gradient = tf.reduce_max(tf.abs(grad), axis=-1)

        # convert to numpy
        gradient = gradient.numpy()

        # # normaliz between 0 and 255
        min_val, max_val = np.min(gradient), np.max(gradient)
        smap = (gradient - min_val) / (max_val - min_val)

        smaps.append(np.int8(smap*255))

    return smaps, results
    # # supress the channel dimension
    # grayscale_tensor = tf.reduce_sum(tf.abs(grad), axis=-1)
    # # normalizaion
    # normalized_tensor = tf.cast(255 * (grayscale_tensor - tf.reduce_min(grayscale_tensor)) / (
    #     tf.reduce_max(grayscale_tensor) - tf.reduce_min(grayscale_tensor)), tf.uint8)
    # normalized_tensor = tf.squeeze(normalized_tensor)

    # smaps.append(normalized_tensor)


def saliency_graph_cut(imgs, saliency_map, background_quantile, foreground_quantile):
    return graph_cut(imgs, saliency_map, background_quantile, foreground_quantile)
