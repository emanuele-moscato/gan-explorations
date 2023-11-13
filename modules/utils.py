import tensorflow as tf


def preprocess_image(image, MAX_VALUE=128.):
    """
    Standardizes the pixel vlaues of images.
    """
    return (tf.cast(image, dtype=tf.float32) - MAX_VALUE) / MAX_VALUE
