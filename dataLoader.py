import tensorflow as tf

MONET_FILENAMES = tf.io.gfile.glob('./dataset/monet_tfrec/*.tfrec')
PHOTO_FILENAMES = tf.io.gfile.glob('./dataset/photo_tfrec/*.tfrec')

IMAGE_SIZE = [256, 256]


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


if __name__ == '__main__':
    print('Monet TFRecord Files:', len(MONET_FILENAMES))
    print('Photo TFRecord Files:', len(PHOTO_FILENAMES))
    monet_ds, photo_ds = get_dataset()
    print(monet_ds, photo_ds)
