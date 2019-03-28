import numpy as np
import tensorflow as tf

from utils.utils import print_variables
from network.official_code_ops import blur2d, upscale2d
from network.common_ops_same import (
    equalized_dense, equalized_conv2d, upscale2d_conv2d, apply_bias, apply_noise,
    pixel_norm, adaptive_instance_norm,
    torgb, lerp, lerp_clip
)


def g_mapping(z, w_dim, n_mapping, n_broadcast):
    with tf.variable_scope('g_mapping'):
        gain = np.sqrt(2)
        lrmul = 0.01

        # normalize input first
        x = pixel_norm(z)

        # run through mapping network
        for ii in range(n_mapping):
            with tf.variable_scope('Dense{:d}'.format(ii)):
                x = equalized_dense(x, w_dim, gain=gain, lrmul=lrmul)
                x = apply_bias(x, lrmul=lrmul)
                x = tf.nn.leaky_relu(x)

        # broadcast to n_layers
        with tf.variable_scope('Broadcast'):
            x = tf.tile(x[:, np.newaxis], [1, n_broadcast, 1])
    return tf.identity(x, name='dlatents_out')


def synthesis_const_block(res, n_f, w_broadcasted, lod):
    lrmul = 1.0
    batch_size = tf.shape(w_broadcasted)[0]

    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Const'):
            x = tf.get_variable('const', shape=[1, n_f, 4, 4], dtype=tf.float32, initializer=tf.initializers.ones())
            x = tf.tile(x, [batch_size, 1, 1, 1])
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w_broadcasted[:, 0])

        with tf.variable_scope('Conv'):
            x = equalized_conv2d(x, n_f, kernel=3, gain=np.sqrt(2), lrmul=1.0)
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w_broadcasted[:, 1])

    # convert to 3-channel image
    images_out = torgb(x, lod=lod)
    return x, images_out


def synthesis_block(x, res, w0, w1, n_f):
    lrmul = 1.0
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Conv0_up'):
            x = upscale2d_conv2d(x, n_f, kernel=3, gain=np.sqrt(2), lrmul=1.0)
            x = blur2d(x, [1, 2, 1])
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w0)

        with tf.variable_scope('Conv1'):
            x = equalized_conv2d(x, n_f, kernel=3, gain=np.sqrt(2), lrmul=1.0)
            x = apply_noise(x)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
            x = adaptive_instance_norm(x, w1)
    return x


def g_synthesis(w_broadcasted, alpha, resolutions, featuremaps):
    assert len(resolutions) == len(featuremaps)

    # there is 2-layers each in every reolution
    with tf.variable_scope('g_synthesis'):
        lods = list(range(len(resolutions) - 1, -1, -1))

        # initial layer
        res = resolutions[0]
        n_f = featuremaps[0]
        lod = lods[0]
        x, images_out = synthesis_const_block(res, n_f, w_broadcasted, lod)

        # remaining layers
        layer_index = 2
        for res, n_f, lod in zip(resolutions[1:], featuremaps[1:], lods[1:]):
            x = synthesis_block(x, res, w_broadcasted[:, layer_index], w_broadcasted[:, layer_index + 1], n_f)
            img = torgb(x, lod)
            images_out = upscale2d(images_out)
            with tf.variable_scope('Grow_lod%d' % lod):
                images_out = lerp_clip(img, images_out, alpha)

            layer_index += 2
    return tf.identity(images_out, name='images_out')


def generator(z, w_dim, n_mapping, alpha, resolutions, featuremaps, is_training):
    # set parameters
    truncation_psi = 0.7
    truncation_cutoff = 8
    dlatent_avg_beta = 0.995
    style_mixing_prob = 0.9

    dlatent_avg = tf.get_variable('dlatent_avg', shape=[w_dim], initializer=tf.initializers.zeros(), trainable=False)

    if is_training:
        truncation_psi = None
        truncation_cutoff = None
    if not is_training:
        dlatent_avg_beta = None
        style_mixing_prob = None

    # start build layers
    # mapping layers
    n_broadcast = len(resolutions) * 2
    w_broadcasted = g_mapping(z, w_dim, n_mapping, n_broadcast)

    # apply truncation trick
    if truncation_psi is not None and truncation_cutoff is not None:
        with tf.variable_scope('Truncation'):
            layer_idx = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
            ones = np.ones(layer_idx.shape, dtype=np.float32)
            coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
            w_broadcasted = lerp(dlatent_avg, w_broadcasted, coefs)

    # synthesis layers
    images_out = g_synthesis(w_broadcasted, alpha, resolutions, featuremaps)

    return images_out


def main():
    # prepare variables
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    is_training = True
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    alpha = tf.Variable(initial_value=0.0, trainable=False, name='transition_alpha')
    fake_images = generator(z, w_dim, n_mapping, alpha, resolutions, featuremaps, is_training)

    print_variables()
    return


if __name__ == '__main__':
    main()
