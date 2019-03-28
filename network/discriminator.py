import numpy as np
import tensorflow as tf

from network.official_code_ops import blur2d, downscale2d, minibatch_stddev_layer
from network.common_ops import (
    equalized_dense, equalized_conv2d, conv2d_downscale2d, apply_bias,
    lerp_clip
)


def fromrgb(x, res, n_f):
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('FromRGB'):
            x = equalized_conv2d(x, fmaps=n_f, kernel=1, gain=np.sqrt(2), lrmul=1.0)
            x = apply_bias(x, lrmul=1.0)
            x = tf.nn.leaky_relu(x)
    return x


def discriminator_block(x, res, n_f0, n_f1):
    gain = np.sqrt(2)
    lrmul = 1.0
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        with tf.variable_scope('Conv0'):
            x = equalized_conv2d(x, n_f0, kernel=3, gain=gain, lrmul=lrmul)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)

        with tf.variable_scope('Conv1_down'):
            x = blur2d(x, [1, 2, 1])
            x = conv2d_downscale2d(x, n_f1, kernel=3, gain=gain, lrmul=lrmul)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
    return x


def discriminator_last_block(x, res, n_f0, n_f1):
    gain = np.sqrt(2)
    lrmul = 1.0
    with tf.variable_scope('{:d}x{:d}'.format(res, res)):
        x = minibatch_stddev_layer(x, group_size=4, num_new_features=1)
        with tf.variable_scope('Conv0'):
            x = equalized_conv2d(x, n_f0, kernel=3, gain=gain, lrmul=lrmul)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
        with tf.variable_scope('Dense0'):
            x = equalized_dense(x, n_f1, gain=gain, lrmul=lrmul)
            x = apply_bias(x, lrmul=lrmul)
            x = tf.nn.leaky_relu(x)
        with tf.variable_scope('Dense1'):
            x = equalized_dense(x, 1, gain=1.0, lrmul=lrmul)
            x = apply_bias(x, lrmul=lrmul)
    return x


def discriminator(image, alpha, resolutions, featuremaps):
    assert len(resolutions) == len(featuremaps)

    # discriminator's (resolutions and featuremaps) are reversed against generator's
    r_resolutions = resolutions[::-1]
    r_featuremaps = featuremaps[::-1]

    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        # set inputs
        img = image
        x = fromrgb(image, r_resolutions[0], r_featuremaps[0])

        # stack discriminator blocks
        for index, (res, n_f) in enumerate(zip(r_resolutions[:-1], r_featuremaps[:-1])):
            res_next = r_resolutions[index + 1]
            n_f_next = r_featuremaps[index + 1]

            x = discriminator_block(x, res, n_f, n_f_next)
            img = downscale2d(img)
            y = fromrgb(img, res_next, n_f_next)

            # smooth transition
            with tf.variable_scope('{:d}x{:d}'.format(res, res)):
                with tf.variable_scope('smooth_transition'):
                    x = lerp_clip(x, y, alpha)

        # last block
        res = r_resolutions[-1]
        n_f = r_featuremaps[-1]
        discriminator_last_block(x, res, n_f, n_f)

        scores_out = tf.identity(x, name='scores_out')
    return scores_out


def main():
    from utils.utils import print_variables

    # prepare variables
    zero_init = tf.initializers.zeros()

    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    c_res = 1024
    # c_idx = resolutions.index(c_res)
    alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=zero_init, trainable=False)

    fake_images = tf.constant(0.5, dtype=tf.float32, shape=[1, 3, c_res, c_res])
    fake_score = discriminator(fake_images, alpha, resolutions, featuremaps)

    print(fake_score.shape)
    print_variables()
    return


if __name__ == '__main__':
    main()
