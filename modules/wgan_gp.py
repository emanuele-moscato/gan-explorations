import tensorflow as tf
from tensorflow.keras.metrics import Mean


class WGANGP(tf.keras.Model):
    """
    """
    def __init__(
        self,
        critic,
        generator,
        latent_dim,
        critic_steps,
        gp_weight
    ):
        """
        """
        super().__init__()

        self.critic = critic
        self.generator = generator

        self.latent_dim = latent_dim
        self.critic_steps = critic_steps
        self.gp_weight = gp_weight

    def compile(self, c_optimizer, g_optimizer):
        """
        """
        super().compile()

        # Optimizers, separate for the critic and the generator.
        self.c_optimizer = c_optimizer
        self.g_optimizer = g_optimizer

        # Critic's metrics: the wasserstein one, the gradient penalty and the
        # total one (a linear combination of the two).
        self.c_wass_loss_metric = Mean(name='c_wass_loss')
        self.c_gp_metric = Mean(name='c_gp')
        self.c_loss_metric = Mean(name='c_loss')

        # Generator's metric (Wasserstein).
        self.g_loss_metric = Mean(name='g_loss')

    @property
    def metrics(self):
        """
        """
        return [
            self.c_loss_metric,
            self.c_wass_loss_metric,
            self.c_gp_metric,
            self.g_loss_metric
        ]

    def gradient_penalty(self, batch_size, real_images, fake_images):
        """
        """
        # For each (real_image, fake_image) pair, generate a random linear
        # interpolation (pixel-wise, in in tensities).
        alpha = tf.random.uniform(shape=(batch_size, 1, 1, 1))

        interpolated_images = (
            (fake_images - real_images) * alpha
            + real_images
        )

        with tf.GradientTape() as gp_tape:
            # The gradient must watch `interpolated_images` as we'll ge
            # differentiating the critic's predictions w.r.t. these.
            gp_tape.watch(interpolated_images)

            # Generate the critic's predictions over the interpolated images.
            pred = self.critic(interpolated_images, training=True)

        # Compute the gradients w.r.t. each pixel/channel of each interpolated
        # image.
        # Output shape: (batch_size, img_size, img_size, n_channels).
        grads = gp_tape.gradient(pred, [interpolated_images])[0]

        # Compute the norm of the gradient for each interpolated image. The
        # sum acts on the pixels/channels dimensions so as to get a tensor
        # of shape (batch_size,), on which the square root is applied
        # element-wise.
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))

        # Compute the gradient penalty as the mean square difference between
        # the norms (one for each interpolated image) and the value 1. The
        # mean is taken over the interpolated images (`batch_size` terms).
        gp = tf.reduce_mean((norm - 1.0) ** 2)

        return gp

    def train_step(self, real_images):
        """
        """
        # Infer the batch size from the input.
        # Note: use `tf.shape`` instead of the tensor's `.shape` method, as
        #       only the former works at the build time of the computational
        #       graph.
        batch_size = tf.shape(real_images)[0]

        # Perform a set number of training steps for the critic every training
        # step of the generator.
        for _ in range(self.critic_steps):
            # Generate random latent vectors of the appropriate latent
            # dimension from which the generator creates the fake images.
            random_latent_vectors = tf.random.normal(
                shape=(batch_size, self.latent_dim)
            )

            with tf.GradientTape() as tape:
                # Create the fake images with the generator.
                fake_images = self.generator(
                    random_latent_vectors, training=True
                )

                # Make predictions on the fake and real images with the
                # critic.
                fake_predictions = self.critic(fake_images, training=True)
                real_predictions = self.critic(real_images, training=True)

                # Compute the Wasserstein loss (it's just the difference of
                # the mean predictions on the fake and on the real images).
                c_wass_loss = (
                    tf.reduce_mean(fake_predictions)
                    - tf.reduce_mean(real_predictions)
                )

                # Compute the gradient penalty term.
                c_gp = self.gradient_penalty(
                    batch_size, real_images, fake_images
                )

                # Compute the total loss for the critic as a sum of the
                # Wassersten and the gradient penalty one, the latter with a
                # given relative weight.
                c_loss = c_wass_loss + self.gp_weight * c_gp

            # Compute the gradient of the critic's loss w.r.t. its weights.
            c_gradient = tape.gradient(
                c_loss, self.critic.trainable_variables
            )

            # Perform a gradient descent step for the critic's weights.
            self.c_optimizer.apply_gradients(
                zip(c_gradient, self.critic.trainable_variables)
            )

        # Perform a training step for the generator.
        random_latent_vectors = tf.random.uniform(
            shape=(batch_size, self.latent_dim)
        )

        with tf.GradientTape() as tape:
            # Create fake images from the latent vectors with the generator.
            fake_images = self.generator(random_latent_vectors)

            # Compute the critic's predictions for the fake images.
            fake_predictions = self.critic(fake_images)

            # Compute the Wasserstein loss for the generator.
            g_loss = - tf.reduce_mean(fake_predictions)

        # Compute the gradient of the generator's loss w.r.t. to its weights.
        g_gradient = tape.gradient(g_loss, self.generator.trainable_variables)

        # Perform a radient descent step for the generator's weights.
        self.g_optimizer.apply_gradients(
            zip(g_gradient, self.generator.trainable_weights)
        )

        # Update the critic's metrics.
        self.c_loss_metric.update_state(c_loss)
        self.c_wass_loss_metric.update_state(c_wass_loss)
        self.c_gp_metric.update_state(c_gp)

        # Update the generator's metrics.
        self.g_loss_metric.update_state(g_loss)

        # Return a dictionary with metric names and values for the current
        # training step.
        return {metric.name: metric.result() for metric in self.metrics}



