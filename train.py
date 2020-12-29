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

def create_submission_file():
    print('Generating submission.csv file...')
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
    np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',', header='id,label', comments='')
    #!head submission.csv

def to_float32(image, label):
    return tf.cast(image, tf.float32), label

def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label

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

def load_dataset(filenames, labeled=True, ordered=False):
    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return dataset

def get_training_set(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.batch(defs.BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_validation_set(filenames):
    dataset = load_dataset(filenames)
    dataset = dataset.batch(defs.BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

def get_test_set(filenames, ordered):    
    dataset = load_dataset(filenames, labeled=False, ordered=ordered)
    dataset = dataset.batch(defs.BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset

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

    print("Train TFRecord Files:", len(TRAINING_FILENAMES))
    print("Validation TFRecord Files:", len(VALID_FILENAMES))

    def count_data_items(filenames):
        n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
        return np.sum(n)
   
    TEST_FILENAMES = tf.io.gfile.glob(defs.BASE_FOLDER + '/test_tfrecords/ld_test*.tfrec')
    training_set = get_training_set(TRAINING_FILENAMES)
    valid_set = get_validation_set(VALID_FILENAMES)
    testing_dataset = get_test_set(TEST_FILENAMES, ordered=False)
    testing_dataset = testing_dataset.unbatch().batch(20)

    early_stopping_cb = tf.keras.callbacks.EarlyStopping(
        patience=5,
        restore_best_weights=True
    )

    model.model.fit(
            training_set,
            epochs=1,
            validation_data=valid_set,
            callbacks=[early_stopping_cb]
    )

    print("Test TFRecord Files:", len(TEST_FILENAMES))
    testing_set = get_test_set(TEST_FILENAMES, ordered=True) 
    testing_set = testing_set.map(to_float32)
    test_images_ds = testing_set.map(lambda image, idnum: image)
    probabilities = model.model.predict(test_images_ds)
    predictions = np.argmax(probabilities, axis=-1)
    print(predictions)

if __name__ == "__main__":
    model = cnn_model()
    train_from_tf_records(model)