import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, LeakyReLU, Dropout, Flatten,
    Concatenate)


class Critic(tf.keras.Model):
    """
    Subclass of Keras' `Model` implementing the critic (i.e. discriminator
    part) of a WGAN-GP. The architecture is a that of a standard CNN for
    classification, with a sequence of convolutional blocks with an increasing
    number of filters, the only difference being that the final convolutional
    layer has identity (i.e. absent) ativation function, allowing for the
    output to be in the (-infinity, +infinity) range.
    """
    def __init__(self, conv_blocks_filters=[128, 256, 512]):
        """
        Class constructor instantiating the layers of the NN.
        """
        super().__init__()

        # Instantiate the inizial convolutional block (with no dropout
        # regularization).
        self.initial_conv_block = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(0.2)
        ])

        # Instantiate the sequence of convolutional blocks with dropout
        # regularization.
        self.regularized_conv_blocks = [
            tf.keras.Sequential([
                Conv2D(
                    filters=n_filters, kernel_size=4, strides=2, padding='same'
                ),
                LeakyReLU(0.2),
                Dropout(0.3)
            ])
            for n_filters in conv_blocks_filters
        ]

        # Final convolutional layer with identity activation. The
        # convolutional layers are engineered to get a final shape of
        # (1, 1, 1) for each sample.
        self.final_conv_layer = Conv2D(
            filters=1, kernel_size=4, strides=1, padding='valid'
        )

        self.flatten = Flatten()

    def call(self, x):
        """
        Forward pass.
        """
        x = self.initial_conv_block(x)

        for block in self.regularized_conv_blocks:
            x = block(x)

        x = self.final_conv_layer(x)

        x = self.flatten(x)

        return x


class CGANCritic(tf.keras.Model):
    """
    Subclass of Keras' `Model` implementing the critic (i.e. discriminator
    part) of a  conditional WGAN-GP. The architecture is same as in the
    `Critic` model, but here the one-hot encoded labels are an additional part
    of the input and are concatenated to the input image. To do so, before
    being passed to the model, their tensor rank is expanded and the values
    repeated along the new axes.
    """
    def __init__(self, conv_blocks_filters=[128, 256, 512]):
        """
        Class constructor instantiating the layers of the NN.
        """
        super().__init__()

        # Instantiate the first concatenation layer.
        self.concat = Concatenate(axis=-1)

        # Instantiate the first convolutional block (with no dropout
        # regularization).
        self.initial_conv_block = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=4, strides=2, padding='same'),
            LeakyReLU(0.2)
        ])

        # Instantiate the sequence of convolutional blocks with dropout
        # regularization.
        self.regularized_conv_blocks = [
            tf.keras.Sequential([
                Conv2D(
                    filters=n_filters, kernel_size=4, strides=2, padding='same'
                ),
                LeakyReLU(0.2),
                Dropout(0.3)
            ])
            for n_filters in conv_blocks_filters
        ]

        # Final convolutional layer with identity activation. The
        # convolutional layers are engineered to get a final shape of
        # (1, 1, 1) for each sample.
        self.final_conv_layer = Conv2D(
            filters=1, kernel_size=4, strides=1, padding='valid'
        )

        self.flatten = Flatten()

    def call(self, model_input):
        """
        Forward pass.

        Note: `model_input` is assumed to be a list in which
                * 1st entry: image.
                * 2nd entry: reshaped one-hot encoded class label.
        """
        image_input, label_input = model_input

        x = self.concat([image_input, label_input])

        x = self.initial_conv_block(x)

        for block in self.regularized_conv_blocks:
            x = block(x)

        x = self.final_conv_layer(x)

        x = self.flatten(x)

        return x
