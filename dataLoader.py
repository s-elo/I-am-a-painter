import tensorflow as tf
import os
import cv2
import numpy as np

if os.path.exists('./dataset'):
    MONET_FILENAMES = tf.io.gfile.glob('./dataset/monet_tfrec/*.tfrec')
    PHOTO_FILENAMES = tf.io.gfile.glob('./dataset/photo_tfrec/*.tfrec')
else:
    MONET_FILENAMES = tf.io.gfile.glob(
        '../input/gan-getting-started/monet_tfrec/*.tfrec')
    PHOTO_FILENAMES = tf.io.gfile.glob(
        '../input/gan-getting-started/photo_tfrec/*.tfrec')

IMAGE_SIZE = [256, 256]

SALIENCY_PATH = f'./dataset/saliency_imgs'


def decode_image(image):
    image = tf.image.decode_jpeg(image, channels=3)
    image = (tf.cast(image, tf.float32) / 127.5) - 1
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image


def read_tfrecord(example):
    tfrecord_format = {
        "image_name": tf.io.FixedLenFeature([], tf.string),
        "image": tf.io.FixedLenFeature([], tf.string),
        "target": tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example['image'])
    return image


def load_dataset(filenames):
    dataset = tf.data.TFRecordDataset(filenames)
    dataset = dataset.map(
        read_tfrecord, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


def get_dataset():
    monet_ds = load_dataset(MONET_FILENAMES).batch(1)
    photo_ds = load_dataset(PHOTO_FILENAMES).batch(1)

    return (monet_ds, photo_ds)


def imagenet_class_to_idx():
    class_to_idx_path = f'{SALIENCY_PATH}/imagenet_class_to_idx_torch_hub.txt'
    idx_map = {}

    # map the folder name as index order
    with open(class_to_idx_path, 'r') as f:
        for _, line in enumerate(f.readlines()):
            class_name, idx = line.replace('\n', '').split('\t')
            idx_map[class_name] = idx

    return idx_map


def imagenet_idx_to_label():
    idx_to_label_path = f'{SALIENCY_PATH}/imagenet_idx_label.txt'
    label_map = {}

    # map the folder name as index order
    with open(idx_to_label_path, 'r') as f:
        for _, line in enumerate(f.readlines()):
            class_idx, label = line.replace('\n', '').split('  ')
            # just map to the first type of label
            label_map[class_idx] = label.split(',')[0]

    return label_map


def get_saliency_imgs():
    imgs = []
    labels = []

    idx_map = imagenet_class_to_idx()

    classes = os.listdir(SALIENCY_PATH)
    for class_folder in classes:
        class_path = f'{SALIENCY_PATH}/{class_folder}'
        if (os.path.isfile(class_path)):
            continue

        for img_name in os.listdir(class_path):
            img_path = f'{class_path}/{img_name}'

            # get the real image data
            i = cv2.imread(img_path)
            # resize
            i = cv2.resize(i, (224, 224), interpolation=cv2.INTER_CUBIC)
            # reshape
            i = np.reshape(i, (224, 224, 3)) / 255

            imgs.append(i)
            labels.append(idx_map[class_folder])

    return np.array(imgs), np.array(labels)


if __name__ == '__main__':
    print('Monet TFRecord Files:', len(MONET_FILENAMES))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))
    monet_ds, photo_ds = get_dataset()
    print(monet_ds, photo_ds)

    imgs, labels = get_saliency_imgs()
    print(imgs.shape, labels.shape)
    # print(imagenet_class_to_idx())
    # print(imagenet_idx_to_label())
