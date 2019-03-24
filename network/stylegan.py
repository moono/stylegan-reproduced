import numpy as np
import tensorflow as tf

from network.official_code_ops import blur2d, upscale2d, downscale2d, minibatch_stddev_layer


def apply_bias(x, units, lrmul=1.0):
    bias = tf.get_variable('bias', shape=[units], dtype=x.dtype, initializer=tf.initializers.zeros()) * lrmul
    if len(x.shape) == 2:
        x = x + bias
    else:
        x = x + tf.reshape(bias, [1, -1, 1, 1])
    return x


def equalized_dense(x, units, gain=np.sqrt(2), lrmul=1.0, add_bias=True):
    def prepare_weights(in_features, out_features):
        he_std = gain / tf.sqrt(tf.cast(in_features, dtype=tf.float32))  # He init
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
        weight = tf.get_variable('weight', shape=[in_features, out_features], dtype=x.dtype,
                                 initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
        return weight

    with tf.variable_scope('equalized_dense'):
        x = tf.reshape(x, shape=[-1, x.shape[1]])
        w = prepare_weights(x.shape[1], units)
        x = tf.matmul(x, w)
        if add_bias:
            x = apply_bias(x, units, lrmul)
    return x


def equalized_conv2d(x, features, kernels, gain=np.sqrt(2), lrmul=1.0, add_bias=False):
    def prepare_weights(k, in_features, out_features):
        he_std = gain / tf.sqrt(tf.cast(k * k * in_features, dtype=tf.float32))  # He init
        init_std = 1.0 / lrmul
        runtime_coef = he_std * lrmul
        weight = tf.get_variable('weight', shape=[k, k, in_features, out_features], dtype=x.dtype,
                                 initializer=tf.initializers.random_normal(0, init_std)) * runtime_coef
        return weight

    with tf.variable_scope('equalized_conv2d'):
        w = prepare_weights(kernels, x.shape[1], features)
        x = tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME', data_format='NCHW')
        if add_bias:
            x = apply_bias(x, features, lrmul)
    return x


def to_rgb(x):
    with tf.variable_scope('to_rgb'):
        x = equalized_conv2d(x, features=3, kernels=1, gain=1.0, lrmul=1.0, add_bias=True)
    return x


def from_rgb(x, res, n_f):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('from_rgb'):
            x = equalized_conv2d(x, features=n_f, kernels=1, gain=np.sqrt(2), lrmul=1.0, add_bias=True)
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
        style = equalized_dense(w, x.shape[1]*2, gain=1.0, lrmul=1.0, add_bias=True)
        style = tf.reshape(style, [-1, 2, x.shape[1]] + [1] * (len(x.shape) - 2))
        style = x * (style[:, 0] + 1) + style[:, 1]
    return style


# b module
def add_noise(x, noise, add_bias=True):
    with tf.variable_scope('add_noise'):
        channels = x.shape[1]   # tf.shape(x)[1]
        weight = tf.get_variable('weight', shape=[channels], dtype=x.dtype, initializer=tf.initializers.zeros())
        x = x + noise * tf.reshape(weight, [1, -1, 1, 1])

        if add_bias:
            x = apply_bias(x, channels, lrmul=1.0)
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
                x = equalized_dense(x, w_dim, gain=np.sqrt(2), lrmul=0.01, add_bias=True)
                x = tf.nn.leaky_relu(x)
                x = tf.identity(x, name='w')
    return x


def synthesis_block(x, w0, w1, noise_image0, noise_image1, n_f):
    with tf.variable_scope('conv0'):
        x = upscale2d(x)
        x = equalized_conv2d(x, n_f, kernels=3, gain=np.sqrt(2), lrmul=1.0, add_bias=False)
        x = blur2d(x, [1, 2, 1])
        x = add_noise(x, noise_image0, add_bias=True)
        x = tf.nn.leaky_relu(x)
        x = instance_norm(x)
        x = style_mod(x, w0)

    with tf.variable_scope('conv1'):
        x = equalized_conv2d(x, n_f, kernels=3, gain=np.sqrt(2), lrmul=1.0, add_bias=False)
        x = add_noise(x, noise_image1, add_bias=True)
        x = tf.nn.leaky_relu(x)
        x = instance_norm(x)
        x = style_mod(x, w1)
    return x


def synthesis_network(w_broadcast, noise_images, alpha, resolutions, featuremaps):
    dtype = tf.float32  # w.dtype
    batch_size = tf.shape(w_broadcast[0])[0]

    # early layers
    res = resolutions[0]
    n_f = featuremaps[0]
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('const'):
            layer_index = 0
            x = tf.get_variable('const', shape=[1, n_f, 4, 4], dtype=dtype, initializer=tf.initializers.ones())
            x = tf.tile(x, [batch_size, 1, 1, 1])
            x = add_noise(x, noise_images[layer_index], add_bias=True)
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)
            x = style_mod(x, w_broadcast[layer_index])

        with tf.variable_scope('conv'):
            layer_index = 1
            x = equalized_conv2d(x, n_f, kernels=3, gain=np.sqrt(2), lrmul=1.0, add_bias=False)
            x = add_noise(x, noise_images[layer_index], add_bias=True)
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)
            x = style_mod(x, w_broadcast[layer_index])

        # convert to 3-channel image
        prev_img = to_rgb(x)

    # remaning layers
    layer_index = 2
    for res, n_f in zip(resolutions[1:], featuremaps[1:]):
        # set systhesis block
        with tf.variable_scope('{:d}x{:d}'.format(res, res)):
            x = synthesis_block(x, w_broadcast[layer_index], w_broadcast[layer_index + 1],
                                noise_images[layer_index], noise_images[layer_index + 1], n_f)
            img = to_rgb(x)
            prev_img = upscale2d(prev_img)

            # smooth transition
            with tf.variable_scope('smooth_transition'):
                prev_img = img + (prev_img - img) * tf.clip_by_value(alpha, 0.0, 1.0)

        # update layer index
        layer_index += 2

    image_out = tf.identity(prev_img, name='image_out')
    return image_out


def update_moving_average_of_w(w_broadcasted, w_dim, moving_avg_beta):
    with tf.variable_scope('moving_average_w'):
        w_moving_avg = tf.get_variable('w_moving_avg', shape=[w_dim], initializer=tf.initializers.zeros(),
                                       trainable=False)
        w_link = w_broadcasted[0]
        batch_avg = tf.reduce_mean(w_link, axis=0)
        update_op = tf.assign(w_moving_avg, batch_avg + (w_moving_avg - batch_avg) * moving_avg_beta)
        with tf.control_dependencies([update_op]):
            w_link = tf.identity(w_link)
    return


def style_mixing(z, w_dim, n_mapping, n_layers):
    with tf.name_scope('StyleMix'):
        z2 = tf.random_normal(tf.shape(z))
        w2 = mapping_network(z, w_dim, n_mapping)
        layer_idx = np.arange(n_layers)[np.newaxis, :, np.newaxis]
        cur_layers = n_layers - tf.cast(lod_in, tf.int32) * 2
        mixing_cutoff = tf.cond(
            tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
            lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
            lambda: cur_layers)
        dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents)), dlatents, dlatents2)
    return


def generator(z, w_dim, n_mapping, alpha, resolutions, featuremaps, is_training):
    moving_avg_beta = 0.995     # decay for tracking the moving average of W during training
    style_mixing_prob = 0.9     # probability of mixing styles during training
    if not is_training:
        moving_avg_beta = None
        style_mixing_prob = None

    with tf.variable_scope('generator'):
        # prepare inputs
        dtype = tf.float32  # w.dtype
        batch_size = tf.shape(z)[0]
        n_layers = len(resolutions) * 2

        # disentangled latents: w
        # run through mapping network and broadcast to n_layers
        w = mapping_network(z, w_dim, n_mapping)
        w_broadcasted = w_broadcaster(w, w_dim, n_layers)

        # update moving average of w
        if moving_avg_beta is not None:
            update_moving_average_of_w(w_broadcasted, w_dim, moving_avg_beta)

        # Perform style mixing regularization.
        if style_mixing_prob is not None:
            style_mixing()

        # create noise images: noise
        noise_images = list()
        for res in resolutions:
            noise_shape = [batch_size, 1, res, res]
            with tf.variable_scope('{:d}x{:d}'.format(res, res)):
                noise_images.append(tf.random_normal(noise_shape, dtype=dtype, name='noise_0'))
                noise_images.append(tf.random_normal(noise_shape, dtype=dtype, name='noise_1'))

        with tf.variable_scope('synthesis'):
            fake_images = synthesis_network(w_broadcasted, noise_images, alpha, resolutions, featuremaps)
    return fake_images


def discriminator_block(x, res, n_f, n_f_next):
    lrmul = 1.0
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('conv0'):
            x = equalized_conv2d(x, n_f, kernels=3, gain=np.sqrt(2), lrmul=lrmul, add_bias=True)
            x = tf.nn.leaky_relu(x)

        with tf.variable_scope('conv1'):
            x = blur2d(x, [1, 2, 1])
            x = equalized_conv2d(x, n_f_next, kernels=3, gain=np.sqrt(2), lrmul=lrmul, add_bias=False)
            x = downscale2d(x)
            x = apply_bias(x, n_f_next, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
    return x


def discriminator(image, alpha, resolutions, featuremaps):
    assert len(resolutions) == len(featuremaps)

    # discriminator's (resolutions and featuremaps) are reversed against generator's
    r_resolutions = resolutions[::-1]
    r_featuremaps = featuremaps[::-1]

    with tf.variable_scope('discriminator'):
        # set inputs
        img = image
        x = from_rgb(image, r_resolutions[0], r_featuremaps[0])

        # stack discriminator blocks
        for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
            res_next = r_resolutions[index + 1]
            n_f_next = r_featuremaps[index + 1]

            x = discriminator_block(x, res, n_f, n_f_next)
            img = downscale2d(img)
            y = from_rgb(img, res_next, n_f_next)

            # smooth transition
            with tf.variable_scope('smooth_transition'):
                x = x + (y - x) * tf.clip_by_value(alpha, 0.0, 1.0)

        # last block
        lrmul = 1.0
        res = r_resolutions[-1]
        n_f = r_featuremaps[-1]
        with tf.variable_scope('{:d}x{:d}'.format(res, res)):
            x = minibatch_stddev_layer(x, group_size=4, num_new_features=1)
            with tf.variable_scope('conv0'):
                x = equalized_conv2d(x, n_f, kernels=3, gain=np.sqrt(2), lrmul=lrmul, add_bias=True)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('dense1'):
                x = equalized_dense(x, n_f, gain=np.sqrt(2), lrmul=1.0, add_bias=True)
                x = tf.nn.leaky_relu(x)
            with tf.variable_scope('dense2'):
                x = equalized_dense(x, 1, gain=1.0, lrmul=1.0, add_bias=True)
                x = tf.nn.leaky_relu(x)

        scores_out = tf.identity(x, name='scores_out')
    return scores_out


def print_variables():
    import pprint

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


def main():
    # prepare generator variables
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    is_training = True
    # final_output_resolution = 1024
    # resolution_log2 = int(np.log2(float(final_output_resolution)))
    # resolutions = [2 ** (power + 1) for power in range(1, resolution_log2)]
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    print(resolutions)
    print(featuremaps)

    r_resolutions = resolutions[::-1]
    r_featuremaps = featuremaps[::-1]
    for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
        print(index, res, n_f)

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    alpha = tf.Variable(initial_value=0.0, trainable=False, name='transition_alpha')
    fake_images = generator(z, w_dim, n_mapping, alpha, resolutions, featuremaps, is_training)
    d_score = discriminator(fake_images, alpha, resolutions, featuremaps)

    print_variables()

    return


if __name__ == '__main__':
    main()
