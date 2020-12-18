import pandas as pd
from tensorflow.keras import preprocessing
from tensorflow.data import Dataset
import tensorflow as tf
from cnn import cnn_model
import definitions as defs
import numpy as np
import os


def train():

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
        im_id = parts[-2]
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


if __name__ == "__main__":
    train()