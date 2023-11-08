import tensorflow as tf
from tensorflow.keras.layers import (Reshape, Conv2DTranspose,
    BatchNormalization, LeakyReLU)


GENERATOR_CONV_DEFAULT_PARAMS = [
    {
        'filters': 512,
        'kernel_size': 4,
        'strides': 1,
        'padding': 'valid',
        'use_bias': False
    },
    {
        'filters': 256,
        'kernel_size': 4,
        'strides': 2,
        'padding': 'same',
        'use_bias': False
    },
    {
        'filters': 128,
        'kernel_size': 4,
        'strides': 2,
        'padding': 'same',
        'use_bias': False
    },
    {
        'filters': 64,
        'kernel_size': 4,
        'strides': 2,
        'padding': 'same',
        'use_bias': False
    }
]


class Generator(tf.keras.Model):
    """
    Subclass of Keras `Model` implementing the generator part of a GAN. The
    model should learn to map a vector randomly sampled from a d-dimensional
    latent space (in this case, d=100) in which datapoints are assumed to have
    a multivariate standard Normal distribution to a realistic image.

    The architecture is based on a stack of blocks with structure
      Transposed convolution -> Batch normalization -> Leaky ReLU,
    which progressvely increase the image size and reduce the channel
    dimension.

    An initial `Reshape` layer turns the input tensor from shape
    (100,) to shape (1, 1, 100) (a 1x1 image with 100 channels), which is then
    fed to the blocks above.

    A final `Conv2DTranspose` layer outputs a tensor with the same shape as
    the images from the dataset, usign a tanh activation function because
    pixel intensities are assumed to be normalized in [-1, 1] (normalization
    could have been performed in [0, 1], in which case a sigmoid activation
    would have been used, but tanh is apparently more robust against
    exploding/vanishing gradients).
    """
    def __init__(
        self,
        latent_dim=100,
        normalized_conv_blocks_params=GENERATOR_CONV_DEFAULT_PARAMS,
        output_channels=1
    ):
        """
        Class constructor for the generator model.

        Parameters
        ----------
        latent_dim : int, optional (default: 1)
            Dimension of the latent space. Each input sample is assumed to
            have shape (latent_dim,).
        normalized_conv_blocks_params : list, optional
                (default: GENERATOR_CONV_DEFAULT_PARAMS)
            List of dictionaries containing the parameters of the
            `Conv2DTranspose` layer within each block, ordered as they should
            appear in the sequence. The number of blocks is inferred from the
            length of the list.
        output_channels : int, optional (default: 1)
            Number of output channels. The default value 1 corresponds to
            grayscale.
        """
        super().__init__()

        # Instantiate (blocks of) layers.
        self.reshape = Reshape((1, 1, latent_dim))

        self.normalized_conv_blocks = [
            tf.keras.Sequential([
                Conv2DTranspose(**params),
                BatchNormalization(momentum=0.9),
                LeakyReLU(0.2)
            ])
            for params in normalized_conv_blocks_params
        ]

        # Instantiate the final `Conv2DTranspose` layer. The output shape
        # should be (n, n, 1), where n is the size of the images in the
        # dataset. tanh activation is used because pixel intensities are
        # assumed to be normalized in [-1, 1].
        self.output_conv = Conv2DTranspose(
            filters=output_channels,
            kernel_size=4,
            strides=2,
            padding='same',
            use_bias=False,
            activation='tanh'
        )

    def call(self, x):
        """
        Forward pass.
        """
        # Reshape the input tensor from (latent_dim,) to (1, 1, latent_dim).
        x = self.reshape(x)

        # Apply each transpose convolution block sequentially.
        for normalized_conv_block in self.normalized_conv_blocks:
            x = normalized_conv_block(x)

        # Apply the final transpose convolution layer.
        x = self.output_conv(x)

        return x
