import pandas as pd
from tensorflow.keras import preprocessing
from tensorflow.data import Dataset
import tensorflow as tf
from cnn import cnn_model
import definitions as defs
import numpy as np
import os


def train_from_raw_images():

    batch_size = 4
    training_df = pd.read_csv(defs.BASE_FOLDER + defs.TRAINING_FILENAME)
    train_ds = Dataset.list_files(str(defs.BASE_FOLDER+defs.TRAINING_FOLDER+'*'), shuffle=False)
    for f in train_ds.take(5):
        print(f.numpy())
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    class_names = np.array([0,1,2,3,4])
    print(class_names)

    def get_label(file_path):
        print(file_path)
        parts = tf.strings.split(file_path, os.path.sep)
        print(parts)
        im_id = parts[-2] == training_df['image_id'].astype(str)
        print("Image id: ", im_id)
        print(training_df['image_id'] == im_id)
        return training_df['image_id'] == im_id

    def decode_img(img):
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [defs.IMAGE_HEIGHT, defs.IMAGE_WIDTH])
        
    def process_path(file_path):
        label = get_label(file_path)
        img = tf.io.read_file(file_path)
        img = decode_img(img)

    # Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
    train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)

    for image, label in train_ds.take(1):
        print("Image shape: ", image.numpy().shape)
        print("Label: ", label.numpy())


def train_from_tf_records():
    TRAINING_FILENAMES, VALID_FILENAMES = train_test_split(
        tf.io.gfile.glob(defs.BASE_FOLDER + '/train_tfrecords/ld_train*.tfrec'),
        test_size=0.35, random_state=5
    )

    TEST_FILENAMES = tf.io.gfile.glob(defs.BASE_FOLDER + '/test_tfrecords/ld_test*.tfrec')

    print("Train TFRecord Files:", len(TRAINING_FILENAMES))
    print("Validation TFRecord Files:", len(VALID_FILENAMES))
    print("Test TFRecord Files:", len(TEST_FILENAMES))

    image_feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'target': tf.io.FixedLenFeature([], tf.int64),
    } if labeled else {
        'image': tf.io.FixedLenFeature([], tf.string),
        'image_name': tf.io.FixedLenfeature([], tf.string)
    }

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
        dataset = load_dataset(TEST_FILENAMES)
        dataset = dataset.batch(defs.BATCH_SIZE)
        dataset = dataset.prefetch(AUTOTUNE)
        return dataset


if __name__ == "__main__":
    train_from_tf_records()