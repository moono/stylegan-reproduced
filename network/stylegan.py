import numpy as np
import tensorflow as tf


def equalized_dense(x, units, gain=np.sqrt(2), lrmul=1.0):
    def prepare_weights(in_features, out_features):
        he_std = gain / np.sqrt(in_features)  # He init
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul

        weight = tf.get_variable('weight', shape=[in_features, out_features], dtype=x.dtype,
                                 initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
        bias = tf.get_variable('bias', shape=[out_features], dtype=x.dtype,
                               initializer=tf.initializers.zeros()) * lrmul
        return weight, bias

    with tf.variable_scope('equalized_dense'):
        x = tf.layers.flatten(x)
        w, b = prepare_weights(x.get_shape().as_list()[1], units)
        x = tf.matmul(x, w) + b
    return x


def equalized_conv2d(x, features, kernels, gain=np.sqrt(2), lrmul=1.0):
    def prepare_weights(k, in_features, out_features):
        he_std = gain / np.sqrt(k * k * in_features)  # He init
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul

        weight = tf.get_variable('weight', shape=[k, k, in_features, out_features], dtype=x.dtype,
                                 initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
        return weight

    with tf.variable_scope('equalized_conv2d'):
        w = prepare_weights(kernels, x.get_shape().as_list()[1], features)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    return x


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('pixel_norm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)
    return x


# a module: adaptive instance norm 1
def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('instance_norm'):
        orig_dtype = x.dtype
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')

        x = tf.cast(x, tf.float32)
        x = x - tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        x = x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
    return x


# a module: adaptive instance norm 2
def style_mod(x, w):
    with tf.variable_scope('style_mod'):
        style = equalized_dense(w, x.shape[1]*2, gain=1.0, lrmul=1.0)
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        style = x * (style[:, 0] + 1) + style[:, 1]
    return style


# b module
def add_noise(x, noise, scope_name):
    with tf.variable_scope(scope_name):
        channels = x.shape[1]   # tf.shape(x)[1]
        weight = tf.get_variable('weight', shape=[1, channels, 1, 1], dtype=x.dtype,
                                 initializer=tf.initializers.zeros())
        b = tf.get_variable('bias', shape=[1, channels, 1, 1], initializer=tf.initializers.zeros())
        x = x + noise * weight + b
    return x


def mapping_network(z, z_dim=512, w_dim=512, n_mapping=8):
    # prepare inputs
    x = tf.convert_to_tensor(z)
    x.set_shape([None, z_dim])

    # normalize latents
    x = pixel_norm(x)

    # run through mapping network
    for ii in range(n_mapping):
        with tf.variable_scope('layer_{:d}'.format(ii)):
            x = equalized_dense(x, w_dim, gain=np.sqrt(2), lrmul=0.01)
            x = tf.nn.leaky_relu(x)
            x = tf.identity(x, name='w')
    return x


def synthesis_network(w_broadcast, w_dim, noise_images=None, resolutions=None):
    dtype = tf.float32  # w.dtype
    batch_size = tf.shape(w_broadcast[0])[0]
    n_styles = len(resolutions) * 2  # if use_styles else 1

    # early layers
    with tf.variable_scope('4x4'):
        with tf.variable_scope('const'):
            layer_index = 0
            x = tf.get_variable('const', shape=[1, w_dim, 4, 4], dtype=dtype, initializer=tf.initializers.ones())
            x = tf.tile(x, [batch_size, 1, 1, 1])
            x = add_noise(x, noise_images[layer_index], scope_name='noise_{:d}'.format(layer_index))
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)
            x = style_mod(x, w_broadcast[layer_index])

        with tf.variable_scope('conv'):
            layer_index = 1
            x = equalized_conv2d(x, w_dim, kernels=3, gain=np.sqrt(2), lrmul=0.01)
            x = add_noise(x, noise_images[layer_index], scope_name='noise_{:d}'.format(layer_index))
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)
            x = style_mod(x, w_broadcast[layer_index])
    return x


def style_generator(z, z_dim, w_dim, n_mapping, resolutions, random_noise=True):
    # prepare inputs to synthesis network
    n_layers = len(resolutions) * 2

    # disentangled latents: w
    # run through mapping network and broadcast to n_layers
    with tf.variable_scope('mapping'):
        w = mapping_network(z, z_dim, w_dim, n_mapping)

        w_broadcast = list()
        with tf.variable_scope('broadcast'):
            w_tiled = tf.tile(w[:, np.newaxis], [1, n_layers, 1])
            for layer_index in range(n_layers):
                w_broadcast.append(
                    tf.reshape(w_tiled[:, layer_index], shape=[-1, w_dim], name='w_{:d}'.format(layer_index)))

    # noise images: noise
    dtype = tf.float32  # w.dtype
    batch_size = tf.shape(w)[0]
    noise_images = list()
    for ii in range(n_layers):
        res = ii // 2 + 2
        noise_image_size = 2 ** res
        name = 'noise_{:d}'.format(ii)
        if random_noise:
            noise_shape = [batch_size, 1, noise_image_size, noise_image_size]
            noise_images.append(tf.random_normal(noise_shape, dtype=dtype, name=name))
        else:
            noise_shape = [1, 1, noise_image_size, noise_image_size]
            noise_images.append(tf.get_variable(name, shape=noise_shape, dtype=dtype,
                                                initializer=tf.initializers.random_normal(), trainable=False))

    with tf.variable_scope('synthesis'):
        fake_images = synthesis_network(w_broadcast, w_dim, noise_images, resolutions)
    return fake_images


def nf(stage):
    fmap_base = 8192
    fmap_decay = 1.0
    fmap_max = 512
    return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


def main():
    out1 = nf(1)
    out2 = nf(2)
    out3 = nf(3)
    out4 = nf(4)

    # prepare generator variables
    z_dim = 512
    w_dim = 512
    random_noise = True
    n_mapping = 8
    final_output_resolution = 1024
    resolution_log2 = int(np.log2(final_output_resolution))
    resolutions = [2 ** (power + 1) for power in range(1, resolution_log2)]

    z = tf.placeholder(tf.float32, shape=[None, 512], name='z')
    fake_images = style_generator(z, z_dim, w_dim, n_mapping, resolutions, random_noise)
    return


if __name__ == '__main__':
    main()
