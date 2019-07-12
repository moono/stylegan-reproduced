import numpy as np
import tensorflow as tf

from network.official_code_ops import blur2d, upscale2d
from network.common_ops import (
    dense, conv2d, upscale2d_conv2d, apply_bias, apply_noise,
    pixel_norm, adaptive_instance_norm, lerp, torgb, smooth_transition
)


def g_mapping(z, w_dim, n_mapping, n_broadcast):
    with tf.variable_scope('g_mapping', reuse=tf.AUTO_REUSE):
        gain = np.sqrt(2)
        lrmul = 0.01

        # normalize input first
        x = pixel_norm(z)

        # run through mapping network
        for ii in range(n_mapping):
            with tf.variable_scope('Dense{:d}'.format(ii)):
                x = dense(x, w_dim, gain=gain, lrmul=lrmul)
                x = apply_bias(x, lrmul=lrmul)
                x = tf.nn.leaky_relu(x)

        # broadcast to n_layers
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])
    return tf.identity(x, name='w_broadcasted')


# initial synthesis block: const input
def synthesis_const_block(res, w0, w1, n_f):
    gain = np.sqrt(2)
    lrmul = 1.0
    batch_size = tf.shape(w0)[0]

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, n_f, 4, 4], dtype=tf.float32, initializer=tf.initializers.ones())
            x = tf.tile(x, [batch_size, 1, 1, 1])
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w0)

        with tf.variable_scope('Conv'):
            x = conv2d(x, n_f, kernel=3, gain=gain, lrmul=lrmul)
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w1)
    return x


def synthesis_block(x, res, w0, w1, n_f):
    gain = np.sqrt(2)
    lrmul = 1.0
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Conv0_up'):
            x = upscale2d_conv2d(x, n_f, kernel=3, gain=gain, lrmul=lrmul)
            x = blur2d(x, [1, 2, 1])
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w0)

        with tf.variable_scope('Conv1'):
            x = conv2d(x, n_f, kernel=3, gain=gain, lrmul=lrmul)
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w1)
    return x


def g_synthesis(w_broadcasted, alpha, resolutions, featuremaps):
    # use smooth transition on only current training resolution
    transition_res = resolutions[-1]

    # there is 2-layers each in every resolution
    with tf.variable_scope('g_synthesis', reuse=tf.AUTO_REUSE):

        # initial layer
        res = resolutions[0]
        n_f = featuremaps[0]
        x = synthesis_const_block(res, w_broadcasted[:, 0], w_broadcasted[:, 1], n_f)
        images_out = torgb(x, res=res)

        # remaining layers
        layer_index = 2
        for res, n_f in zip(resolutions[1:], featuremaps[1:]):
            x = synthesis_block(x, res, w_broadcasted[:, layer_index], w_broadcasted[:, layer_index + 1], n_f)
            img = torgb(x, res)
            images_out = upscale2d(images_out)
            images_out = smooth_transition(images_out, img, res, transition_res, alpha)

            layer_index += 2
    return tf.identity(images_out, name='images_out')


def update_moving_average_of_w(w_broadcasted, w_avg, w_ema_decay):
    with tf.variable_scope('wAvg'):
        batch_avg = tf.reduce_mean(w_broadcasted[:, 0], axis=0)
        update_op = tf.assign(w_avg, lerp(batch_avg, w_avg, w_ema_decay))
        with tf.control_dependencies([update_op]):
            w_broadcasted = tf.identity(w_broadcasted)
    return w_broadcasted


def style_mixing_regularization(z, w_dim, w_broadcasted, n_mapping, n_broadcast,
                                train_res_block_idx, style_mixing_prob):
    with tf.name_scope('StyleMix'):
        z2 = tf.random_normal(tf.shape(z), dtype=tf.float32)
        w_broadcasted2 = g_mapping(z2, w_dim, n_mapping, n_broadcast)
        layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
        last_layer_index = (train_res_block_idx + 1) * 2
        mixing_cutoff = tf.cond(
            tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
            lambda: tf.random_uniform([], 1, last_layer_index, dtype=tf.int32),
            lambda: tf.constant(last_layer_index, dtype=tf.int32))
        w_broadcasted = tf.where(tf.broadcast_to(layer_indices < mixing_cutoff, tf.shape(w_broadcasted)),
                                 w_broadcasted,
                                 w_broadcasted2)
    return w_broadcasted


def truncation_trick(n_broadcast, w_broadcasted, w_avg, truncation_psi, truncation_cutoff):
    with tf.variable_scope('Truncation'):
        layer_indices = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
        ones = np.ones(layer_indices.shape, dtype=np.float32)
        coefs = tf.where(layer_indices < truncation_cutoff, truncation_psi * ones, ones)
        w_broadcasted = lerp(w_avg, w_broadcasted, coefs)
    return w_broadcasted


def generator(z, alpha, g_params, is_training):
    # set parameters
    w_dim = g_params['w_dim']
    n_mapping = g_params['n_mapping']
    resolutions = g_params['resolutions']
    featuremaps = g_params['featuremaps']
    w_ema_decay = g_params.get('w_ema_decay', None)
    style_mixing_prob = g_params.get('style_mixing_prob', None)
    truncation_psi = g_params.get('truncation_psi', None)
    truncation_cutoff = g_params.get('truncation_cutoff', None)
    train_res_block_idx = len(resolutions) - 1

    # check input parameters
    assert len(resolutions) == len(featuremaps)
    assert len(resolutions) >= 2

    tf.summary.scalar('alpha_generator', alpha)

    # more variables
    with tf.variable_scope('', reuse=tf.AUTO_REUSE):
        w_avg = tf.get_variable('w_avg', shape=[w_dim], dtype=tf.float32, initializer=tf.initializers.zeros(),
                                trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

    # start building layers
    # mapping layers
    n_broadcast = len(resolutions) * 2
    w_broadcasted = g_mapping(z, w_dim, n_mapping, n_broadcast)

    # apply regularization techniques on training
    if is_training:
        # update moving average of w
        w_broadcasted = update_moving_average_of_w(w_broadcasted, w_avg, w_ema_decay)

        # perform style mixing regularization
        w_broadcasted = style_mixing_regularization(z, w_dim, w_broadcasted, n_mapping, n_broadcast,
                                                    train_res_block_idx, style_mixing_prob)

    # apply truncation trick on evaluation
    if not is_training:
        w_broadcasted = truncation_trick(n_broadcast, w_broadcasted, w_avg, truncation_psi, truncation_cutoff)

    # synthesis layers
    images_out = g_synthesis(w_broadcasted, alpha, resolutions, featuremaps)

    return images_out


def test_generator_network(resolutions, featuremaps):
    from utils.utils import print_variables

    # prepare variables
    is_training = True
    z_dim = 512
    g_params = {
        'w_dim': 512,
        'n_mapping': 8,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
        'w_ema_decay': 0.995,
        'style_mixing_prob': 0.9,
        'truncation_psi': 0.7,
        'truncation_cutoff': 8,
    }

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=tf.initializers.zeros(), trainable=False)

    fake_images = generator(z, alpha, g_params, is_training)
    print('output fake image shape: {}'.format(fake_images.shape))

    print_variables()
    return


def main():
    # original
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    test_generator_network(resolutions, featuremaps)

    # # reduced
    # resolutions = [4, 8]
    # featuremaps = [512, 512]
    # test_generator_network(resolutions, featuremaps)
    return


if __name__ == '__main__':
    main()
