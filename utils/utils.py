import pprint
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
