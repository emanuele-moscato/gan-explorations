import os
import datetime
import sys
import tensorflow as tf


sys.path.append('../modules/')


from wgan_gp_critic import Critic
from generator import Generator
from wgan_gp import WGANGP
from utils import preprocess_image, select_model_logs_dir, DATETIME_FORMAT
from logger import get_logger


DATA_DIR = '../data/dataset/'
LOGS_DIR = '../logs/'

LATEST_PID = 2956


if __name__ == '__main__':
    logger = get_logger('gan_logger')

    logger.info('Starting')

    logger.info('Getting training data')

    # Load data.
    training_data = tf.keras.utils.image_dataset_from_directory(
        directory=DATA_DIR,
        labels=None,
        color_mode='grayscale',
        batch_size=128,
        image_size=(64, 64),
    )

    # training_data = training_data.repeat()

    # Preprocess the data.
    training_data = training_data.map(lambda img: preprocess_image(img))

    # Create model.
    logger.info('Instantiating model')

    critic = Critic()
    generator = Generator()

    wgangp_model = WGANGP(
        critic=critic,
        generator=generator,
        latent_dim=100,
        critic_steps=3,
        gp_weight=10.
    )

    # Create empty `History` object (with training history data to be
    # appended to it).
    full_history = tf.keras.callbacks.History()

    # Create a model logs directory for Tensorboard.
    model_logs_dir = select_model_logs_dir(
        LOGS_DIR, append_to_latest_logs=False
    )

    # Create tensorflow callback.
    tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=model_logs_dir
    )

    wgangp_model.compile(
        c_optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999),
        g_optimizer=tf.keras.optimizers.Adam(beta_1=0.9, beta_2=0.999)
    )

    # Train model.
    logger.info('Training starting')

    wgangp_model.fit(
        training_data,
        epochs=20,
        steps_per_epoch=None,
        callbacks=[tensorboard_callback]
    )

    logger.info('Training finished')

    logger.info('Saving model')

    model_name = model_logs_dir[19:-1]

    wgangp_model.save(f'../models/{model_name}.keras')

    logger.info('Model saved')



    # Test.
    # model_dropout_loaded = tf.keras.models.load_model(
    #     f'../models/{model_name}.keras',
    #     safe_mode=False
    # )

