from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import definitions as defs

class cnn_model():
    def __init__(self):
        self.build_model()

    def build_model(self):

        IMAGE_SIZE = [512, 512]

        img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input,
                                                  input_shape=[*IMAGE_SIZE, 3])
    
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        base_model.trainable = False
    
        self.model = tf.keras.Sequential([
            tf.keras.layers.BatchNormalization(renorm=True),
            img_adjust_layer,
            base_model,
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(8, activation='relu'),
            #tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Dense(5, activation='softmax')  
        ])

        lr_scheduler = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=1e-5, 
            decay_steps=10000, 
            decay_rate=0.9)
        
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr_scheduler, epsilon=0.001),
            loss='sparse_categorical_crossentropy',  
            metrics=['sparse_categorical_accuracy']
        )
        print("model built")

    def train(self):
        self.model.fit()