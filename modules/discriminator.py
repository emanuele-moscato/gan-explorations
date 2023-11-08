import tensorflow as tf
from tensorflow.keras.layers import (Conv2D, LeakyReLU, Dropout,
    BatchNormalization, Flatten)


class Discriminator(tf.keras.Model):
    """
    Subclass of Keras `Model` implementing the discriminator part of a GAN.
    The model has an image as its input and outputs the predicted probability
    for the image to be real (i.e. to have been drawn from the real dataset
    rather than having been created by the generator part of the network). The
    architecture is a stack (sequence) of blocks with the structure
      Convolution -> (Batch normalization) -> Leaky ReLU activation ->
      -> Dropout regularization,
    with the number of filters progressively increasing and the dimension of
    the image progressively decreasing. Convolutions have been engineered to
    obtain an output with shape (1, 1, 1) in the end, so no final
    fully-connected block is needed. A single value in [0, 1] is outputted
    (using a final sigmoid activation) for each sample.
    """
    def __init__(
        self,
        conv_blocks_filters=[64],
        normalized_conv_blocks_filters=[128, 256, 512]
    ):
        """
        Class constructor for the discriminator model.

        Parameters
        ----------
        conv_blocks_filters : list, optional (default: [64])
            List of integers representing the number of filters to use in the
            `Conv2D` layer of each convolution block. The number of blocks to
            instantiate is inferred from the length of the list.
        normalized_conv_blocks_filters : list, optional
                (default: [128, 256, 512])
            List of integers representing the number of filters to use in the
            `Conv2D` layer of each convolution block with batch normalization.
            The number of blocks to instantiate is inferred from the length of
            the list.
        """
        super().__init__()

        # Instantiate convolution blocks.
        self.conv_blocks = [
            tf.keras.Sequential([
                Conv2D(
                    n_filters,
                    kernel_size=4,
                    strides=2,
                    padding='same',
                    use_bias=False
                ),
                LeakyReLU(0.2),
                Dropout(0.3),
            ])
            for n_filters in conv_blocks_filters
        ]

        # Instantiate convolution blocks with batch normalization.
        self.normalized_conv_blocks = [
            tf.keras.Sequential([
                Conv2D(
                    n_filters,
                    kernel_size=4,
                    strides=2,
                    padding='same',
                    use_bias=False
                ),
                BatchNormalization(momentum=0.9),
                LeakyReLU(0.2),
                Dropout(0.3)
            ])
            for n_filters in normalized_conv_blocks_filters
        ]

        # Final convolution layer.
        # Output shape: (1, 1, 1).
        self.final_conv = Conv2D(
            1,
            kernel_size=4,
            strides=1,
            padding='valid',
            use_bias=False,
            activation='sigmoid'
        )

        # Final flattening layer, converting the output of the final
        # convolution from a tensor with shape (1, 1, 1) to a tensor with
        # shape (1,).
        self.flatten = Flatten()

    def call(self, x):
        """
        Forward pass.
        """
        for conv_block in self.conv_blocks:
            x = conv_block(x)

        for normalized_conv_block in self.normalized_conv_blocks:
            x = normalized_conv_block(x)

        x = self.final_conv(x)

        x = self.flatten(x)

        return x
