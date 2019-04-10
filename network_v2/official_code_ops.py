import numpy as np
import tensorflow as tf


# ----------------------------------------------------------------------------
# Primitive ops for manipulating 4D activation tensors.
# The gradients of these are not necessary efficient or even meaningful.
def _blur2d(x, f, normalize=True, flip=False, stride=1):
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

def blur2d(x, f, normalize=True):
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


def minibatch_stddev_layer(x, group_size=4, num_new_features=1):
    with tf.variable_scope('MinibatchStddev'):
        group_size = tf.minimum(group_size, tf.shape(x)[0])
        s = x.shape
        y = tf.reshape(x, [group_size, -1, num_new_features, s[1] // num_new_features, s[2], s[3]])
        y = tf.cast(y, tf.float32)
        y -= tf.reduce_mean(y, axis=0, keepdims=True)
        y = tf.reduce_mean(tf.square(y), axis=0)
        y = tf.sqrt(y + 1e-8)
        y = tf.reduce_mean(y, axis=[2, 3, 4], keepdims=True)
        y = tf.reduce_mean(y, axis=[2])
        y = tf.cast(y, x.dtype)
        y = tf.tile(y, [group_size, 1, s[2], s[3]])
        return tf.concat([x, y], axis=1)


# def training_schedule(cur_nimg,
#                       lod_training_kimg=600,
#                       lod_transition_kimg=600,
#                       resolution_log2=10,
#                       lod_initial_resolution=8,
#                       minibatch_base=32,
#                       minibatch_dict=None,
#                       num_gpus=1,
#                       max_minibatch_per_gpu=None,
#                       lrate_rampup_kimg=0,
#                       g_lrate_base=0.001,
#                       d_lrate_base=0.001,
#                       g_lrate_dict=None,
#                       d_lrate_dict=None):
#     if minibatch_dict is None:
#         # minibatch_dict = {4: 512, 8: 256, 16: 128, 32: 64, 64: 32}
#         minibatch_dict = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4, 512: 2}
#     if max_minibatch_per_gpu is None:
#         max_minibatch_per_gpu = dict()
#     if g_lrate_dict is None:
#         g_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
#     if d_lrate_dict is None:
#         d_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
#
#     s = dict()
#     s['kimg'] = cur_nimg / 1000.0
#
#     # Training phase.
#     phase_dur = lod_training_kimg + lod_transition_kimg
#     phase_idx = int(np.floor(s['kimg'] / phase_dur)) if phase_dur > 0 else 0
#     phase_kimg = s['kimg'] - phase_idx * phase_dur
#
#     # Level-of-detail and resolution.
#     s['lod'] = resolution_log2
#     s['lod'] -= np.floor(np.log2(lod_initial_resolution))
#     s['lod'] -= phase_idx
#     if lod_transition_kimg > 0:
#         s['lod'] -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
#     s['lod'] = max(s['lod'], 0.0)
#     s['resolution'] = 2 ** (resolution_log2 - int(np.floor(s['lod'])))
#
#     # Minibatch size.
#     s['minibatch'] = minibatch_dict.get(s['resolution'], minibatch_base)
#     s['minibatch'] -= s['minibatch'] % num_gpus
#     if s['resolution'] in max_minibatch_per_gpu:
#         s['minibatch'] = min(s['minibatch'], max_minibatch_per_gpu[s['resolution']] * num_gpus)
#
#     # Learning rate.
#     s['G_lrate'] = g_lrate_dict.get(s['resolution'], g_lrate_base)
#     s['D_lrate'] = d_lrate_dict.get(s['resolution'], d_lrate_base)
#     if lrate_rampup_kimg > 0:
#         rampup = min(s['kimg'] / lrate_rampup_kimg, 1.0)
#         s['G_lrate'] *= rampup
#         s['D_lrate'] *= rampup
#     return s
