from tensorflow.keras import backend as K
from tensorflow.keras import models
from tensorflow.keras import layers

class LeNet:

    @staticmethod
    def build(height, width, channels, classes):

        input_shape = (height, width, channels)
        if K.image_data_format() == "channels_first":
            input_shape = (channels, height, width)

        model = models.Sequential()

        model.add(layers.Conv2D(20, (5, 5), padding="same", input_shape=input_shape))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Conv2D(50, (5, 5), padding="same"))
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(2, 2)))

        model.add(layers.Flatten())
        model.add(layers.Dense(500, activation="relu"))
        model.add(layers.Dense(classes, activation="softmax"))

        return model
