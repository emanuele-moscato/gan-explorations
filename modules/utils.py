import os
import datetime
import numpy as np
import tensorflow as tf


DATETIME_FORMAT = '%Y%m%d_%H%M%S'


def get_latest_model_logs_dir(logs_dir, datetime_format='%Y%m%d_%H%M%S'):
    """
    Given the directory `logs_dir` containing the directories with the logs
    for the specific models, returns the path of the model logs directory
    corresponding to the latest model.

    Model logs directories are assumed to have the path
            `{logs_dir}/logs_model_{model_datetime}/'
    where `{model_datetime}' is in the form specified by the `datetime_format`
    argument.
    """
    # Get all model logs directories.
    model_logs_dirs = os.listdir(logs_dir)

    # Get all timestamps associated with model logs.
    model_timestamps = [
        datetime.datetime.strptime(dirname[11:], datetime_format)
        for dirname in model_logs_dirs
    ]

    # Get the model log directory associated with the latest timestamp.
    max_timestamp = max(model_timestamps)

    latest_model_logs_dir = [
        dirname for dirname in model_logs_dirs
        if (datetime.datetime.strptime(dirname[11:], DATETIME_FORMAT)
            == max_timestamp)
    ][0]

    # Return path to the log directory associated with the latest timestamp.
    return os.path.join(logs_dir, latest_model_logs_dir + '/')


def select_model_logs_dir(logs_dir, append_to_latest_logs=False, logger=None):
    """
    Returns the path to the model logs directory for Tensorboard logs, placed
    under the path specified in `logs_dir`. If `append_to_latest_logs` is
    `False`, a new directory is created, named using the current timestamp,
    otherwise the directory corresponding to the latest timestamp among the
    ones in `logs_dir` is returned.
    """
    if append_to_latest_logs:
        # Fetch the path of the model logs directory corresponding to the
        # latest timestamp.
        model_logs_dir = get_latest_model_logs_dir(logs_dir)

        message = f'Using latest found model logs directory: {model_logs_dir}'

        if logger is not None:
            logger.info(message)
        else:
            print(message)
    else:
        # Create a directory for Tensorboard log corresponding to the current
        # datetime.
        current_datetime = datetime.datetime.strftime(
            datetime.datetime.now(), format=DATETIME_FORMAT
        )

        model_logs_dir = os.path.join(
            logs_dir,
            f"logs_model_{current_datetime}/"
        )

        if not os.path.exists(model_logs_dir):
            os.makedirs(model_logs_dir)

            message = f'Created model logs directory: {model_logs_dir}'

            if logger is not None:
                logger.info(message)
            else:
                print(message)
        else:
            message = (
                f'WARNING: model logs directory {model_logs_dir} already '
                'existing'
            )

            if logger is not None:
                logger.info(message)
            else:
                print(message)

    return model_logs_dir


def save_model_with_opt_state(
        model,
        path,
        optimizer_names=['optimizer']
    ):
    """
    Saves the specified model to `path`, along with the variables of all the
    listed optimizers the model contains.
    """
    model.save(path)

    print(f'Model saved to: {path}')

    for opt_name in optimizer_names:
        opt = getattr(model, opt_name)

        opt_vars_path = (
            '/'.join(path.split('/')[:-1])
            + '/'
            + path.split('/')[-1].split('.')[0]
            + f'_{opt_name}_variables.npy'
        )

        np.save(
            opt_vars_path,
            np.array([v.numpy() for v in opt.variables], dtype=object),
            allow_pickle=True
        )

        print(f'Saved variables of optimizer {opt_name} to: {opt_vars_path}')


def preprocess_image(image, MAX_VALUE=128.):
    """
    Rescale pixel intensities (single channel - grayscale)
    to be in the [-1, 1] interval.
    """
    return (tf.cast(image, dtype=tf.float32) - MAX_VALUE) / MAX_VALUE


def inverse_preprocessing(gen_output, MAX_VALUE=128.):
    """
    Perform the inverse operation w.r.t. the preprocessing
    step, so that the generator's output's pixel intensities
    are converted to the usual [0, 127] scale.

    Note: it is assumed that the generator outputs pixel intensities in the
          [-1, 1] range (e.g. by using tanh activation in the last layer).
    """
    return tf.cast(gen_output * MAX_VALUE + MAX_VALUE, tf.int32)
