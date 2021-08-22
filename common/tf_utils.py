import tensorflow as tf


def set_memory_growth(mode):
    physical_devices = tf.config.list_physical_devices('GPU')

    for physical_device in physical_devices:
        tf.config.experimental.set_memory_growth(physical_device, mode)
