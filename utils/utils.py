import pprint
import numpy as np
import tensorflow as tf


def print_variables():
    t_vars = tf.trainable_variables()
    nt_vars = list()
    for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
        if not v.trainable:
            nt_vars.append(v)

    print('Non-Trainable')
    pprint.pprint(nt_vars)

    print('Trainable')
    pprint.pprint(t_vars)
    return


def post_process_generator_output(generator_output):
    generator_output_shape = generator_output.shape

    drange_min, drange_max = -1.0, 1.0
    scale = 255.0 / (drange_max - drange_min)

    if len(generator_output_shape) == 4:
        scaled_image = np.squeeze(generator_output, axis=0)
    elif len(generator_output_shape) == 3:
        scaled_image = generator_output
    else:
        raise ValueError('generator output image shape should be [1, 3, height, width] or [3, height, width]')
    scaled_image = np.transpose(scaled_image, axes=[1, 2, 0])
    scaled_image = scaled_image * scale + (0.5 - drange_min * scale)
    scaled_image = np.clip(scaled_image, 0, 255)
    scaled_image = scaled_image.astype('uint8')
    return scaled_image
