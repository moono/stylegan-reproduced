import numpy as np
import tensorflow as tf

from network.official_code_ops import blur2d, upscale2d


def equalized_dense(x, units, gain=np.sqrt(2), lrmul=1.0):
    def prepare_weights(in_features, out_features):
        # he_std = gain / np.sqrt(in_features)  # He init
        he_std = gain / tf.sqrt(tf.to_float(in_features))  # He init
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul

        weight = tf.get_variable('weight', shape=[in_features, out_features], dtype=x.dtype,
                                 initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
        bias = tf.get_variable('bias', shape=[out_features], dtype=x.dtype,
                               initializer=tf.initializers.zeros()) * lrmul
        return weight, bias

    with tf.variable_scope('equalized_dense'):
        x = tf.layers.flatten(x)
        # w, b = prepare_weights(x.get_shape().as_list()[1], units)
        w, b = prepare_weights(x.shape[1], units)
        x = tf.matmul(x, w) + b
    return x


def equalized_conv2d(x, features, kernels, gain=np.sqrt(2), lrmul=1.0):
    def prepare_weights(k, in_features, out_features):
        # he_std = gain / np.sqrt(k * k * in_features)  # He init
        he_std = gain / tf.sqrt(tf.to_float(k * k * in_features))  # He init
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul

        weight = tf.get_variable('weight', shape=[k, k, in_features, out_features], dtype=x.dtype,
                                 initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
        return weight

    with tf.variable_scope('equalized_conv2d'):
        # w = prepare_weights(kernels, x.get_shape().as_list()[1], features)
        w = prepare_weights(kernels, x.shape[1], features)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
    return x


def to_rgb(x):
    with tf.variable_scope('to_rgb'):
        x = equalized_conv2d(x, features=3, kernels=1, gain=1.0, lrmul=1.0)
    return x


def from_rgb(x, n_features):
    with tf.variable_scope('from_rgb'):
        x = equalized_conv2d(x, features=n_features, kernels=1, gain=1.0, lrmul=1.0)
        x = tf.nn.leaky_relu(x)
    return x


def lerp_clip(a, b, t):
    with tf.name_scope("lerp_clip"):
        lerp_cliped = a + (b - a) * tf.clip_by_value(t, 0.0, 1.0)
    return lerp_cliped


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
def add_noise(x, noise):
    with tf.variable_scope('add_noise'):
        channels = x.shape[1]   # tf.shape(x)[1]
        weight = tf.get_variable('weight', shape=[1, channels, 1, 1], dtype=x.dtype,
                                 initializer=tf.initializers.zeros())
        b = tf.get_variable('bias', shape=[1, channels, 1, 1], initializer=tf.initializers.zeros())
        x = x + noise * weight + b
    return x


def w_broadcaster(w, w_dim, n_layers):
    w_broadcast = list()
    with tf.variable_scope('broadcast'):
        w_tiled = tf.reshape(w, shape=[-1, 1, w_dim])
        w_tiled = tf.tile(w_tiled, [1, n_layers, 1])
        for layer_index in range(n_layers):
            current_layer = w_tiled[:, layer_index]
            current_layer = tf.reshape(current_layer, shape=[-1, w_dim], name='w_{:d}'.format(layer_index))
            w_broadcast.append(current_layer)
    return w_broadcast


def mapping_network(z, w_dim=512, n_mapping=8):
    with tf.variable_scope('mapping'):
        # normalize latents
        x = pixel_norm(z)

        # run through mapping network
        for ii in range(n_mapping):
            with tf.variable_scope('layer_{:d}'.format(ii)):
                x = equalized_dense(x, w_dim, gain=np.sqrt(2), lrmul=0.01)
                x = tf.nn.leaky_relu(x)
                x = tf.identity(x, name='w')
    return x


def synthesis_block(x, w0, w1, noise_image0, noise_image1, n_features):
    with tf.variable_scope('conv0'):
        x = upscale2d(x)
        x = equalized_conv2d(x, n_features, kernels=3, gain=np.sqrt(2), lrmul=1.0)
        x = blur2d(x, [1, 2, 1])
        x = add_noise(x, noise_image0)
        x = tf.nn.leaky_relu(x)
        x = instance_norm(x)
        x = style_mod(x, w0)

    with tf.variable_scope('conv1'):
        x = equalized_conv2d(x, n_features, kernels=3, gain=np.sqrt(2), lrmul=1.0)
        x = add_noise(x, noise_image1)
        x = tf.nn.leaky_relu(x)
        x = instance_norm(x)
        x = style_mod(x, w1)
    return x


def synthesis_network(w_broadcast, noise_images=None, resolutions=None, featuremaps=None):
    dtype = tf.float32  # w.dtype
    batch_size = tf.shape(w_broadcast[0])[0]

    # early layers
    with tf.variable_scope('4x4'):
        n_features = featuremaps[0]
        with tf.variable_scope('const'):
            layer_index = 0
            x = tf.get_variable('const', shape=[1, n_features, 4, 4], dtype=dtype, initializer=tf.initializers.ones())
            x = tf.tile(x, [batch_size, 1, 1, 1])
            x = add_noise(x, noise_images[layer_index])
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)
            x = style_mod(x, w_broadcast[layer_index])

        with tf.variable_scope('conv'):
            layer_index = 1
            x = equalized_conv2d(x, n_features, kernels=3, gain=np.sqrt(2), lrmul=1.0)
            x = add_noise(x, noise_images[layer_index])
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)
            x = style_mod(x, w_broadcast[layer_index])

        # convert to 3-channel image
        images_out = to_rgb(x)

    # remaning layers
    layer_index = 2
    for res, n_features in zip(resolutions[1:], featuremaps[1:]):
        # set systhesis block
        with tf.variable_scope('{:d}x{:d}'.format(res, res)):
            x = synthesis_block(x, w_broadcast[layer_index], w_broadcast[layer_index + 1],
                                noise_images[layer_index], noise_images[layer_index + 1], n_features)
            img = to_rgb(x)
            images_out = upscale2d(images_out)
            with tf.variable_scope('grow'):
                images_out = lerp_clip(img, images_out, lod_in - lod)

        # update layer index
        layer_index += 2
    return x


def style_generator(z, w_dim, n_mapping, resolutions, featuremaps):
    # prepare inputs
    dtype = tf.float32  # w.dtype
    batch_size = tf.shape(z)[0]
    n_layers = len(resolutions) * 2

    # disentangled latents: w
    # run through mapping network and broadcast to n_layers
    w = mapping_network(z, w_dim, n_mapping)
    w_broadcasted = w_broadcaster(w, w_dim, n_layers)

    # create noise images: noise
    noise_images = list()
    for res in resolutions:
        noise_shape = [batch_size, 1, res, res]
        with tf.variable_scope('{:d}x{:d}'.format(res, res)):
            noise_images.append(tf.random_normal(noise_shape, dtype=dtype, name='noise_0'))
            noise_images.append(tf.random_normal(noise_shape, dtype=dtype, name='noise_1'))

    with tf.variable_scope('synthesis'):
        fake_images = synthesis_network(w_broadcasted, noise_images, resolutions, featuremaps)
    return fake_images


def main():
    # prepare generator variables
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    # final_output_resolution = 1024
    # resolution_log2 = int(np.log2(float(final_output_resolution)))
    # resolutions = [2 ** (power + 1) for power in range(1, resolution_log2)]
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    print(resolutions)
    print(featuremaps)

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    fake_images = style_generator(z, w_dim, n_mapping, resolutions, featuremaps)
    return


if __name__ == '__main__':
    main()
