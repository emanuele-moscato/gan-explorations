import tensorflow as tf
from tensorflow.keras.metrics import Mean, BinaryAccuracy


class DCGAN(tf.keras.Model):
    """
    Subclass of Keras' `Model` implementing a deep convolutional generative
    adversarial network (DCGAN). The training step simultaneously trains the
    discriminator and the generator (although in an independent way).
    """
    def __init__(self, discriminator, generator, latent_dim, noise_param=0.1):
        """
        Class constructor.

        Parameters
        ----------
        discriminator : tf.keras.Model
            Discriminator model.
        generator : tf.keras.Model
            Generator model.
        latent_dim : int
            Number of dimensions of the latent space (i.e. one vector in
            latent space has shape (latent_dim,)).
        noise_param : float, optional (default: 0.1)
            Coefficient for the random (uniform) noise added to the class
            labels to train the discriminator.
        """
        super().__init__()

        self.discriminator = discriminator
        self.generator = generator
        self.latent_dim = latent_dim
        self.noise_param = noise_param

    def compile(self, d_optimizer, g_optimizer):
        """
        Compile method for the DCGAN model: sets the input optimizers as
        instance attributes and instantiates the metrics.
        """
        super().compile()

        # Discriminator's and generator's optimizers.
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer

        # Loss function.
        self.loss_fn = tf.keras.losses.BinaryCrossentropy()

        # Discriminator metrics.
        self.d_loss_metric = Mean(name='d_loss')
        self.d_real_acc_metric = BinaryAccuracy(name='d_real_acc')
        self.d_fake_acc_metric = BinaryAccuracy(name='d_fake_acc')
        self.d_acc_metric = BinaryAccuracy(name='d_acc')

        # Generator metrics.
        self.g_loss_metric = Mean(name='g_loss')
        self.g_acc_metric = BinaryAccuracy(name='g_acc')

    @property
    def metrics(self):
        return [
            self.d_loss_metric,
            self.d_real_acc_metric,
            self.d_fake_acc_metric,
            self.d_acc_metric,
            self.g_loss_metric,
            self.g_acc_metric
        ]

    def train_step(self, real_images):
        """
        A single training step for the GAN model. Both the discriminator and
        the generator's weights are updated (but in an independent way).

        Steps:
          1. Generate fake images (as many as in the input batch) by randomly
             sampling the latent space and passing the corresponding latent
             vectors through the generator.
          2. Get predictions from the discriminator for the real and the fake
             images.
          3. Create labels: all 1's for the real images and all 0's for the
             fake ones.
          4. Separately compute two losses:
               4a. The discriminator's loss is the binary cross entropy
                   obtained using the predictions and labels for the real and
                   fake images, as they are (separate values are computed for
                   real and fake images, and then averaged).
               4b. The generator's loss is the binary cross entropy between
                   the discriminator's predictions on the fake images and the
                   labels for the REAL ones: this measures how far the fake
                   images are from real ones according to the discriminator
                   (the generator's weights will be updated to minimize this
                   "distance").
          5. Separately compute two gradients:
               5a. Gradient of the discriminator's loss w.r.t. the
                   discriminator's weights.
               5b. Gradient of the generator's loss w.r.t. the generator's
                   weights.
          6. Separately apply the gradients.
          7. Update the values of the metrics and return a dictionary
             containing them.

        Notes:
          * The discriminator's loss is computed w.r.t. to labels for the real
            and fake images with some added random noise.
          * When getting tensors' shapes during training, `tf.shape` needs to
            be called on the tensors rather than using the tensor's `shape`
            attribute. This is because `tf.shape` allows for lazy evaluation
            while the class attribute doesn't and when the computation graph
            is built the batch size is still unknown (indeed below
            `real_images.shape[0]` reaturns `None` for the batch size the
            first time it's evaluated, raising an error downstream).

        Parameters
        ----------
        real_images : tensor
            Tensor corresponding to a batch of images.
            Shape: (batch_size, N, N, 1) (for NxN images in gray scale).

        Returns
        -------
        dict
            Dictionary with structure: {metric_name: updated_metric_value}.
        """
        # Sample the latent space randomly (number of samples equal to the
        # size of the batch of real images provided).
        batch_size = tf.shape(real_images)[0]

        random_latent_vectors = tf.random.normal(
            shape=(batch_size, self.latent_dim)
        )

        # Train the discriminator on fake images.
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake images from the randomly sampled latent vectors
            # using the generator.
            generated_images = self.generator(
                random_latent_vectors, training=True
            )

            # Get predictions from the discriminator for the real and the fake
            # images.
            real_predictions = self.discriminator(real_images, training=True)

            fake_predictions = self.discriminator(
                generated_images, training=True
            )

            # Create labels for the real and the fake images.
            real_labels = tf.ones_like(real_predictions)
            fake_labels = tf.zeros_like(fake_predictions)

            # Create noisy versions of the labels by adding uniform noise.
            real_noisy_labels = (
                real_labels
                + self.noise_param * tf.random.uniform(
                    tf.shape(real_predictions))
            )
            fake_noisy_labels = (
                fake_labels
                + self.noise_param * tf.random.uniform(
                    tf.shape(fake_predictions))
            )

            # Compute losses.
            # Discriminator's loss for real images.
            d_real_loss = self.loss_fn(real_noisy_labels, real_predictions)

            # Discriminator's loss for fake images.
            d_fake_loss = self.loss_fn(fake_noisy_labels, fake_predictions)

            # The discriminator's total loss is the average between the one
            # for real and the one for fake images.
            d_loss = (d_real_loss + d_fake_loss) / 2.0

            # Generator's loss: we compare the discriminator's predictions on
            # the generated images with the real labels (i.e. all 1's). The
            # higher this loss, the further the discriminator thinks the
            # generated images are from realistic images.
            g_loss = self.loss_fn(real_labels, fake_predictions)

        # Compute gradients for the disciminator's and the generator's losses.
        d_grad = disc_tape.gradient(
            d_loss, self.discriminator.trainable_variables
        )
        g_grad = gen_tape.gradient(
            g_loss, self.generator.trainable_variables
        )

        # Separately update the weights for the discriminator and the
        # generator.
        self.d_optimizer.apply_gradients(
            zip(d_grad, self.discriminator.trainable_variables)
        )
        self.g_optimizer.apply_gradients(
            zip(g_grad, self.generator.trainable_variables)
        )

        # Update the discriminator's metrics.
        self.d_loss_metric.update_state(d_loss)
        self.d_real_acc_metric.update_state(real_labels, real_predictions)
        self.d_fake_acc_metric.update_state(fake_labels, fake_predictions)
        self.d_acc_metric.update_state(
            [real_labels, fake_labels], [real_predictions, fake_predictions]
        )

        # Update the generator's metrics.
        self.g_loss_metric.update_state(g_loss)
        self.g_acc_metric.update_state(real_labels, fake_predictions)

        # Return a dictionary of metric names and values.
        return {m.name: m.result() for m in self.metrics}
