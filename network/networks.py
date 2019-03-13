import numpy as np
import tensorflow as tf


def pixel_norm(x, epsilon=1e-8):
    epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


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

    x = tf.convert_to_tensor(x)
    x = tf.layers.flatten(x)
    w, b = prepare_weights(x.get_shape().as_list()[1], units)
    x = tf.matmul(x, w) + b
    return x


def mapping_network(x, z_dim=512, w_dim=512, n_level_layers=8, n_mapping=8):
    # prepare inputs
    x = tf.convert_to_tensor(x)
    x.set_shape([None, z_dim])

    # normalize latents
    with tf.variable_scope('pixel_norm'):
        x = pixel_norm(x)

    # run through mapping network
    for ii in range(n_mapping):
        with tf.variable_scope('equalized_dense_{:02d}'.format(ii)):
            x = equalized_dense(x, w_dim, gain=np.sqrt(2), lrmul=1.0)
            x = tf.nn.leaky_relu(x)

    # broadcast
    with tf.variable_scope('broadcast_w'):
        x = tf.tile(x[:, np.newaxis], [1, n_level_layers, 1])

    return tf.identity(x, name='w')


# def nf(stage):
#     return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
#
#
# def blur(x):
#     return blur2d(x, blur_filter) if blur_filter else x


# Things to do at the end of each layer.
def layer_epilogue(x, layer_idx):
    if use_noise:
        x = apply_noise(x, noise_inputs[layer_idx], randomize_noise=randomize_noise)
    x = apply_bias(x)
    x = act(x)
    if use_pixel_norm:
        x = pixel_norm(x)
    if use_instance_norm:
        x = instance_norm(x)
    if use_styles:
        x = style_mod(x, dlatents_in[:, layer_idx], use_wscale=use_wscale)
    return x


def synthesis_network(x, w_dim=512, output_resolution=1024):
    resolution_log2 = int(np.log2(output_resolution))
    n_layers = resolution_log2 * 2 - 2
    n_styles = n_layers     # if use_styles else 1

    # inputs
    x.set_shape([None, n_styles, w_dim])
    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)

    # Noise inputs.
    noise_inputs = []
    for ii in range(n_layers):
        res = ii // 2 + 2
        shape = [1, 1, 2 ** res, 2 ** res]
        noise_inputs.append(
            tf.get_variable('noise_{:02d}'.format(ii), shape=shape, initializer=tf.initializers.random_normal(),
                            trainable=False))

    return


def main():
    z_inputs = tf.placeholder(tf.float32, shape=[None, 512], name='z')
    w = mapping_network(z_inputs)
    print()
    return


if __name__ == '__main__':
    main()
