from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.optimizers import Adam


class cnn_model():
    def __init__(self):
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        act = LeakyReLU(alpha=0.01)
        self.model.add(Conv2D(filters=3,
                              kernel_size=(20, 20),
                              strides=(10, 10),
                              input_shape=(600, 800,3),
                              activation=act))

        self.model.add(Flatten())

        self.model.add(Dense(units=64,
                             activation=act,
                             kernel_initializer='uniform'))

        self.model.add(Dense(units=5,
                             activation='softmax'))

        adam = Adam(lr=0.0008, beta_1=0.9, beta_2=0.999, epsilon=1e-8)

        self.model.compile(optimizer=adam,
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])

    def train(self):
        self.model.fit()