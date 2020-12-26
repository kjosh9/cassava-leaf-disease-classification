import pandas as pd
import re
import tensorflow as tf
import definitions as defs
from sklearn.model_selection import train_test_split
import numpy as np
import os
from functools import partial
from cnn import cnn_model
print("Tensorflow version " + tf.__version__)


AUTOTUNE = tf.data.experimental.AUTOTUNE
IMAGE_SIZE = [512, 512]


def train_from_tf_records(model):

    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        print('Device:', tpu.master())
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    except:
        strategy = tf.distribute.get_strategy()

    TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
        tf.io.gfile.glob(defs.BASE_FOLDER + '/train_tfrecords/ld_train*.tfrec'),
        test_size=0.35, random_state=5
    )

    TEST_FILENAMES = tf.io.gfile.glob(defs.BASE_FOLDER + '/test_tfrecords/ld_test*.tfrec')

    print("Train TFRecord Files:", len(TRAINING_FILENAMES))
    print("Validation TFRecord Files:", len(VALID_FILENAMES))
    print("Test TFRecord Files:", len(TEST_FILENAMES))

    def decode_image(image):
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 255.0
        image = tf.reshape(image, [*IMAGE_SIZE, 3])
        return image

    def read_tfrecord(example, labeled):
        image_feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'target': tf.io.FixedLenFeature([], tf.int64),
        } if labeled else {
            'image': tf.io.FixedLenFeature([], tf.string),
            'image_name': tf.io.FixedLenFeature([], tf.string)
        }
        example = tf.io.parse_single_example(example, image_feature_description)
        image = decode_image(example['image'])
        if labeled:
            label = tf.cast(example['target'], tf.int32)
            return image, label
        idnum = example['image_name']
        return image, idnum

    def data_augment(image, label):
        # Thanks to the dataset.prefetch(AUTO) statement in the following function this happens essentially for free on TPU. 
        # Data pipeline code is executed on the "CPU" part of the TPU while the TPU itself is computing gradients.
        image = tf.image.random_flip_left_right(image)
        return image, label

    def load_dataset(filenames, labeled=True, ordered=False):
        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False # disable order, increase speed
        dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
        dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
        return dataset

    def get_training_set():
        dataset = load_dataset(TRAINING_FILENAMES)
        dataset = dataset.batch(defs.BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def get_validation_set():
        dataset = load_dataset(VALID_FILENAMES)
        dataset = dataset.batch(defs.BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def get_test_set():    
        dataset = load_dataset(TEST_FILENAMES, labeled=False)
        dataset = dataset.batch(defs.BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset

    def count_data_items(filenames):
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)

    NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
    NUM_VALIDATION_IMAGES = count_data_items(VALID_FILENAMES)
    NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)

    print('Dataset: {} training images, {} validation images, {} (unlabeled) test images'.format(
        NUM_TRAINING_IMAGES, NUM_VALIDATION_IMAGES, NUM_TEST_IMAGES))

    print("Training data shapes:")
    for image, label in get_training_set().take(3):
        print(image.numpy().shape, label.numpy().shape)
    print("Training data label examples:", label.numpy())
    print("Validation data shapes:")
    for image, label in get_validation_set().take(3):
        print(image.numpy().shape, label.numpy().shape)
    print("Validation data label examples:", label.numpy())
    print("Test data shapes:")
    for image, idnum in get_test_set().take(3):
        print(image.numpy().shape, idnum.numpy().shape)
    print("Test data IDs:", idnum.numpy().astype('U')) # U=unicode string
    
    training_set = get_training_set()
    valid_set = get_validation_set()

    for image, label in training_set.take(3):
        print(image.numpy().shape, label.numpy().shape)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )

    model.model.fit(
            training_set,
            epochs=2,
            validation_data=valid_set
    )
    
if __name__ == "__main__":
    model = cnn_model()
    train_from_tf_records(model)