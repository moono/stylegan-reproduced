import numpy as np
import tensorflow as tf


# def pixel_norm(x, epsilon=1e-8):
#     epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
#     return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
#
#
# def equalized_dense(x, units, gain=np.sqrt(2), lrmul=1.0):
#     def prepare_weights(in_features, out_features):
#         he_std = gain / np.sqrt(in_features)  # He init
#         init_std = 1.0 / lrmul
#         runtime_coef = he_std * lrmul
#
#         weight = tf.get_variable('weight', shape=[in_features, out_features], dtype=x.dtype,
#                                  initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
#         bias = tf.get_variable('bias', shape=[out_features], dtype=x.dtype,
#                                initializer=tf.initializers.zeros()) * lrmul
#         return weight, bias
#
#     x = tf.convert_to_tensor(x)
#     x = tf.layers.flatten(x)
#     w, b = prepare_weights(x.get_shape().as_list()[1], units)
#     # w, b = prepare_weights(tf.shape(x)[1], units)
#     x = tf.matmul(x, w) + b
#     return x
