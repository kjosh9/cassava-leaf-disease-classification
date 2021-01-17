import tensorflow as tf
import definitions as defs

class cnn_model():
    def __init__(self):
        self.build_model()

    def build_model(self):

        IMAGE_SIZE = [512, 512]

        resize_and_crop = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomCrop(height=512, width=512),
            tf.keras.layers.experimental.preprocessing.Rescaling(1./255)
        ])

        data_augmentation = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.experimental.preprocessing.RandomRotation(0.2),
            tf.keras.layers.experimental.preprocessing.RandomContrast(factor=[1,1])
        ])

        img_adjust_layer = tf.keras.layers.Lambda(tf.keras.applications.resnet50.preprocess_input, input_shape=[*IMAGE_SIZE, 3])
        
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)
        base_model.trainable = True

        self.model = tf.keras.Sequential([
            resize_and_crop,
            data_augmentation,
            tf.keras.layers.BatchNormalization(renorm=True),
            tf.keras.layers.Conv2D(filters=4,
                                kernel_size=(5,5),
                                strides=(2,2),
                                activation='relu',
                                input_shape=[*IMAGE_SIZE,3]),
            tf.keras.layers.Conv2D(filters=4,
                                kernel_size=(4,4),
                                strides=(4,4),
                                activation='relu'),
            tf.keras.layers.Conv2D(filters=4,
                                kernel_size=(4,4),
                                strides=(4,4),
                                activation='relu'),
            tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(8, activation='relu'),
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