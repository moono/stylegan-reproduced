import numpy as np
import tensorflow as tf

from network.official_code_ops import upscale2d, downscale2d


def get_weight(weight_shape, gain, lrmul):
    fan_in = np.prod(weight_shape[:-1])  # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)  # He init

    # equalized learning rate
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    # create variable.
    weight = tf.get_variable('weight', shape=weight_shape, dtype=tf.float32,
                             initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
    return weight


def dense(x, units, gain, lrmul):
    x = tf.layers.flatten(x)
    weight_shape = [x.get_shape().as_list()[1], units]
    w = get_weight(weight_shape, gain, lrmul)
    x = tf.matmul(x, w)
    return x


def conv2d(x, fmaps, kernel, gain, lrmul):
    assert kernel >= 1 and kernel % 2 == 1
    weight_shape = [kernel, kernel, x.get_shape().as_list()[1], fmaps]
    w = get_weight(weight_shape, gain, lrmul)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    return x


# upscale & conv2d for resolution lower than 128
# upscale with conv2d_transpose for reolution higher than 128
def upscale2d_conv2d(x, fmaps, kernel, gain, lrmul):
    batch_size = tf.shape(x)[0]
    height, width = x.shape[2], x.shape[3]
    fused_scale = (min(height, width) * 2) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = upscale2d(x)
        x = conv2d(x, fmaps, kernel, gain, lrmul)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    weight = get_weight([kernel, kernel, x.get_shape().as_list()[1], fmaps], gain, lrmul)
    weight = tf.transpose(weight, [0, 1, 3, 2])  # [kernel, kernel, fmaps_out, fmaps_in]
    weight = tf.pad(weight, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    weight = tf.add_n([weight[1:, 1:], weight[:-1, 1:], weight[1:, :-1], weight[:-1, :-1]])
    output_shape = [batch_size, fmaps, height * 2, width * 2]
    x = tf.nn.conv2d_transpose(x, weight, output_shape, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return x


# conv2d & downscale for resolution lower than 128
# conv2d with downscale for reolution higher than 128
def conv2d_downscale2d(x, fmaps, kernel, gain, lrmul):
    # batch_size = tf.shape(x)[0]
    height, width = x.shape[2], x.shape[3]
    fused_scale = (min(height, width) * 2) >= 128

    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = conv2d(x, fmaps, kernel, gain, lrmul)
        x = downscale2d(x)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d().
    w = get_weight([kernel, kernel, x.get_shape().as_list()[1], fmaps], gain, lrmul)
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]]) * 0.25
    w = tf.cast(w, x.dtype)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return x


def apply_bias(x, lrmul):
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    if len(x.shape) == 2:
        x = x + b
    else:
        x = x + tf.reshape(b, [1, -1, 1, 1])
    return x


def apply_noise(x):
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('Noise'):
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]])
        weight = tf.get_variable('weight', shape=[x.get_shape().as_list()[1]], initializer=tf.initializers.zeros())
        weight = tf.reshape(weight, [1, -1, 1, 1])
        x = x + noise * weight
    return x


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        norm = tf.reduce_mean(tf.square(x), axis=1, keepdims=True)
        x = x * tf.rsqrt(norm + epsilon)
    return x


def instance_norm(x, epsilon=1e-8):
    # x: [?, 512, h, w]
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('InstanceNorm'):
        epsilon = tf.constant(epsilon, dtype=tf.float32, name='epsilon')

        # [?, 512, 1, 1]
        mean = tf.reduce_mean(x, axis=[2, 3], keepdims=True)

        # [?, 512, 1, 1]
        var = tf.reduce_mean(tf.square(x - mean), axis=[2, 3], keepdims=True)

        # [?, 512, h, w]
        x = (x - mean) * tf.rsqrt(var + epsilon)
    return x


def style_mod(x, w):
    # x: [?, 512, h, w]
    # w: [?, 512]
    with tf.variable_scope('StyleMod'):
        # units: 1024
        units = x.shape[1] * 2

        # style: [?, 1024]
        style = dense(w, units, gain=1.0, lrmul=1.0)
        style = apply_bias(style, lrmul=1.0)

        # style: [?, 2, 512, 1, 1]
        style = tf.reshape(style, [-1, 2, x.shape[1], 1, 1])
        scale = style[:, 0]
        bias = style[:, 1]

        # x * (style[:, 0] + 1): [?, 512, h, w]
        # x: [?, 512, h, w]
        x = x * (scale + 1) + bias
    return x


def adaptive_instance_norm(x, w):
    x = instance_norm(x)
    x = style_mod(x, w)
    return x


def lerp(a, b, t):
    # t == 1.0: use b
    # t == 0.0: use a
    with tf.name_scope("Lerp"):
        out = a + (b - a) * t
    return out


def lerp_clip(a, b, t):
    # t >= 1.0: use b
    # t <= 0.0: use a
    with tf.name_scope("LerpClip"):
        out = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return out


def torgb(x, res):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('ToRGB'):
            x = conv2d(x, fmaps=3, kernel=1, gain=1.0, lrmul=1.0)
            x = apply_bias(x, lrmul=1.0)
    return x


def fromrgb(x, res, n_f):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('FromRGB'):
            x = conv2d(x, fmaps=n_f, kernel=1, gain=np.sqrt(2), lrmul=1.0)
            x = apply_bias(x, lrmul=1.0)
            x = tf.nn.leaky_relu(x)
    return x


def smooth_transition(prv, cur, res, transition_res, alpha):
    # alpha == 1.0: use only previous resolution output
    # alpha == 0.0: use only current resolution output

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('smooth_transition'):
            # use alpha for current resolution transition
            if transition_res == res:
                out = lerp_clip(cur, prv, alpha)

            # ex) transition_res=32, current_res=16
            # use res=16 block output
            else:   # transition_res > res
                out = lerp_clip(cur, prv, 0.0)
    return out
