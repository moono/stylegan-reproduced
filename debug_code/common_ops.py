import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.

def _blur2d(x, f=[1, 2, 1], normalize=True, flip=False, stride=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(stride, int) and stride >= 1

    # Finalize filter kernel.
    f = np.array(f, dtype=np.float32)
    if f.ndim == 1:
        f = f[:, np.newaxis] * f[np.newaxis, :]
    assert f.ndim == 2
    if normalize:
        f /= np.sum(f)
    if flip:
        f = f[::-1, ::-1]
    f = f[:, :, np.newaxis, np.newaxis]
    f = np.tile(f, [1, 1, int(x.shape[1]), 1])

    # No-op => early exit.
    if f.shape == (1, 1) and f[0, 0] == 1:
        return x

    # Convolve using depthwise_conv2d.
    orig_dtype = x.dtype
    x = tf.cast(x, tf.float32)  # tf.nn.depthwise_conv2d() doesn't support fp16
    f = tf.constant(f, dtype=x.dtype, name='filter')
    strides = [1, 1, stride, stride]
    x = tf.nn.depthwise_conv2d(x, f, strides=strides, padding='SAME', data_format='NCHW')
    x = tf.cast(x, orig_dtype)
    return x


def _upscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Upscale using tf.tile().
    s = x.shape
    x = tf.reshape(x, [-1, s[1], s[2], 1, s[3], 1])
    x = tf.tile(x, [1, 1, 1, factor, 1, factor])
    x = tf.reshape(x, [-1, s[1], s[2] * factor, s[3] * factor])
    return x


def _downscale2d(x, factor=2, gain=1):
    assert x.shape.ndims == 4 and all(dim.value is not None for dim in x.shape[1:])
    assert isinstance(factor, int) and factor >= 1

    # 2x2, float32 => downscale using _blur2d().
    if factor == 2 and x.dtype == tf.float32:
        f = [np.sqrt(gain) / factor] * factor
        return _blur2d(x, f=f, normalize=False, stride=factor)

    # Apply gain.
    if gain != 1:
        x *= gain

    # No-op => early exit.
    if factor == 1:
        return x

    # Large factor => downscale using tf.nn.avg_pool().
    # NOTE: Requires tf_config['graph_options.place_pruned_graph']=True to work.
    ksize = [1, 1, factor, factor]
    return tf.nn.avg_pool(x, ksize=ksize, strides=ksize, padding='VALID', data_format='NCHW')


# ----------------------------------------------------------------------------
# High-level ops for manipulating 4D activation tensors.
# The gradients of these are meant to be as efficient as possible.

def blur2d(x, f=[1, 2, 1], normalize=True):
    with tf.variable_scope('Blur2D'):
        @tf.custom_gradient
        def func(x):
            y = _blur2d(x, f, normalize)

            @tf.custom_gradient
            def grad(dy):
                dx = _blur2d(dy, f, normalize, flip=True)
                return dx, lambda ddx: _blur2d(ddx, f, normalize)

            return y, grad

        return func(x)


def upscale2d(x, factor=2):
    with tf.variable_scope('Upscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _upscale2d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _downscale2d(dy, factor, gain=factor ** 2)
                return dx, lambda ddx: _upscale2d(ddx, factor)

            return y, grad

        return func(x)


def downscale2d(x, factor=2):
    with tf.variable_scope('Downscale2D'):
        @tf.custom_gradient
        def func(x):
            y = _downscale2d(x, factor)

            @tf.custom_gradient
            def grad(dy):
                dx = _upscale2d(dy, factor, gain=1 / factor ** 2)
                return dx, lambda ddx: _downscale2d(ddx, factor)

            return y, grad

        return func(x)


def get_weight(weight_shape, gain, lrmul):
    print('[get_weight] shape: {}, gain: {:.3f}, lrmul: {}'.format(weight_shape, gain, lrmul))
    fan_in = np.prod(weight_shape[:-1])     # [kernel, kernel, fmaps_in, fmaps_out] or [in, out]
    he_std = gain / np.sqrt(fan_in)         # He init

    # Equalized learning rate and custom learning rate multiplier.
    init_std = 1.0 / lrmul
    runtime_coef = he_std * lrmul

    # Create variable.
    init = tf.initializers.random_normal(0, init_std)
    return tf.get_variable('weight', shape=weight_shape, initializer=init) * runtime_coef


def apply_bias(x, lrmul):
    print('[apply_bias] lrmul: {}'.format(lrmul))
    b = tf.get_variable('bias', shape=[x.shape[1]], initializer=tf.initializers.zeros()) * lrmul
    if len(x.shape) == 2:
        x = x + b
    else:
        x = x + tf.reshape(b, [1, -1, 1, 1])
    return x


def equalized_dense(x, units, gain, lrmul):
    print('[dense] fmaps: {}, gain: {}, lrmul: {}'.format(units, gain, lrmul))
    x = tf.layers.flatten(x)
    weight_shape = [x.shape[1].value, units]
    w = get_weight(weight_shape, gain, lrmul)
    x = tf.matmul(x, w)
    return x


def equalized_conv2d(x, fmaps, kernel, gain, lrmul):
    print('[conv2d] fmaps: {}, kernel: {}, gain: {}, lrmul: {}'.format(fmaps, kernel, gain, lrmul))
    assert kernel >= 1 and kernel % 2 == 1
    weight_shape = [kernel, kernel, x.shape[1].value, fmaps]
    w = get_weight(weight_shape, gain, lrmul)
    x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    return x


def upscale2d_conv2d(x, fmaps, kernel, gain, lrmul):
    print('[upscale2d_conv2d] fmaps: {}, kernel: {}, gain: {}, lrmul: {}'.format(fmaps, kernel, gain, lrmul))
    height, width = x.shape[2], x.shape[3]
    fused_scale = (min(height, width) * 2) >= 128

    print('fused_scale: {}'.format(fused_scale))
    # Not fused => call the individual ops directly.
    if not fused_scale:
        x = upscale2d(x)
        x = equalized_conv2d(x, fmaps, kernel, gain, lrmul)
        return x

    # Fused => perform both ops simultaneously using tf.nn.conv2d_transpose().
    w = get_weight([kernel, kernel, x.shape[1].value, fmaps], gain, lrmul)
    w = tf.transpose(w, [0, 1, 3, 2])  # [kernel, kernel, fmaps_out, fmaps_in]
    w = tf.pad(w, [[1, 1], [1, 1], [0, 0], [0, 0]], mode='CONSTANT')
    w = tf.add_n([w[1:, 1:], w[:-1, 1:], w[1:, :-1], w[:-1, :-1]])
    w = tf.cast(w, x.dtype)
    os = [tf.shape(x)[0], fmaps, x.shape[2] * 2, x.shape[3] * 2]
    x = tf.nn.conv2d_transpose(x, w, os, strides=[1, 1, 2, 2], padding='SAME', data_format='NCHW')
    return x


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('PixelNorm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        norm = tf.reduce_mean(tf.square(x), axis=1, keepdims=True)
        x = x * tf.rsqrt(norm + epsilon)
    return x


def apply_noise(x):
    print('[apply_noise]')
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('Noise'):
        noise = tf.random_normal([tf.shape(x)[0], 1, x.shape[2], x.shape[3]])
        weight = tf.get_variable('weight', shape=[x.shape[1].value], initializer=tf.initializers.zeros())
        weight = tf.reshape(weight, [1, -1, 1, 1])
        x = x + noise * weight
    return x


def instance_norm(x, epsilon=1e-8):
    print('[instance_norm]')
    assert len(x.shape) == 4  # NCHW
    with tf.variable_scope('InstanceNorm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
    return x


def style_mod(x, w):
    print('[style_mod] dlatent: {}'.format(w))
    with tf.variable_scope('StyleMod'):
        units = x.shape[1] * 2
        style = equalized_dense(w, units, gain=1.0, lrmul=1.0)
        style = apply_bias(style, lrmul=1.0)
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        x = x * (style[:, 0] + 1) + style[:, 1]
    return x


def adaptive_instance_norm(x, w):
    x = instance_norm(x)
    x = style_mod(x, w)
    return x


def torgb(x, lod):
    print('[torgb] lod: {}'.format(lod))
    with tf.variable_scope('ToRGB_lod{:d}'.format(lod)):
        x = equalized_conv2d(x, fmaps=3, kernel=1, gain=1.0, lrmul=1.0)
        x = apply_bias(x, lrmul=1.0)
    return x


# def to_rgb(x):
#     with tf.variable_scope('to_rgb'):
#         x = equalized_conv2d(x, fmaps=3, kernel=1, gain=1.0, lrmul=1.0)
#     return x
#
#
# def from_rgb(x, res, n_f):
#     with tf.variable_scope('{:d}x{:d}'.format(res, res)):
#         with tf.variable_scope('from_rgb'):
#             x = equalized_conv2d(x, fmaps=n_f, kernel=1, gain=np.sqrt(2), lrmul=1.0)
#     return x

def lerp(a, b, t):
    """Linear interpolation."""
    with tf.name_scope("Lerp"):
        return a + (b - a) * t


def lerp_clip(a, b, t):
    """Linear interpolation with clip."""
    with tf.name_scope("LerpClip"):
        return a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
