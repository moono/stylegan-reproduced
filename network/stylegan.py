import numpy as np
import tensorflow as tf

# from network.ops import pixel_norm, equalized_dense


def pixel_norm(x, epsilon=1e-8):
    with tf.variable_scope('pixel_norm'):
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
    return x * tf.rsqrt(tf.reduce_mean(tf.square(x), axis=1, keepdims=True) + epsilon)


def instance_norm(x, epsilon=1e-8):
    with tf.variable_scope('instance_norm'):
        orig_dtype = x.dtype
        x = tf.cast(x, tf.float32)
        x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        epsilon = tf.constant(epsilon, dtype=x.dtype, name='epsilon')
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)
        x = tf.cast(x, orig_dtype)
    return x


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
    # w, b = prepare_weights(tf.shape(x)[1], units)
    x = tf.matmul(x, w) + b
    return x


# b module
def add_noise(x, noise, scrop_name):
    with tf.variable_scope(scrop_name):
        channels = tf.shape(x)[1]
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
        with tf.variable_scope('equalized_dense_{:02d}'.format(ii)):
            x = equalized_dense(x, w_dim, gain=np.sqrt(2), lrmul=0.01)
            x = tf.nn.leaky_relu(x)
    return tf.identity(x, name='w')


# def nf(stage, fmap_base, fmap_decay, fmap_max):
#     return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)


def synthesis_network(w, w_dim=512, output_resolution=1024, use_noise=True, random_noise=True):
    dtype = tf.float32  # w.dtype
    batch_size = tf.shape(w)[0]

    resolution_log2 = int(np.log2(output_resolution))
    n_layers = resolution_log2 * 2 - 2
    n_styles = n_layers     # if use_styles else 1

    # # inputs
    # x.set_shape([None, n_styles, w_dim])
    # lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)

    # noise inputs
    noise_inputs = list()
    if use_noise:
        for ii in range(n_layers):
            res = ii // 2 + 2
            name = 'noise_{:02d}'.format(ii)
            if random_noise:
                noise_shape = [batch_size, 1, 2 ** res, 2 ** res]
                noise_inputs.append(tf.random_normal(noise_shape, dtype=dtype, name=name))
            else:
                noise_shape = [1, 1, 2 ** res, 2 ** res]
                noise_inputs.append(tf.get_variable(name, shape=noise_shape, dtype=dtype,
                                                    initializer=tf.initializers.random_normal(), trainable=False))

    # early layers
    with tf.variable_scope('4x4'):
        x = tf.get_variable('const', shape=[1, 512, 4, 4], dtype=dtype, initializer=tf.initializers.ones())
        x = tf.tile(x, [batch_size, 1, 1, 1])
        if use_noise:
            x = add_noise(x, noise_inputs[0], scrop_name='noise_{:02d}'.format(0))
            x = tf.nn.leaky_relu(x)
            x = instance_norm(x)

    return


# def style_generator(z, truncation_psi=0.7, truncation_cutoff=8):
#     # setup variables
#     lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
#     dlatent_avg = tf.get_variable('dlatent_avg', shape=[dlatent_size], initializer=tf.initializers.zeros(),
#                                   trainable=False)
#     return


def generator(z, z_dim, w_dim, n_mapping, n_layers):
    # run through mapping network and broadcast to n_layers
    with tf.variable_scope('mapping'):
        w = mapping_network(z, z_dim, w_dim, n_mapping)
        w = tf.tile(w[:, np.newaxis], [1, n_layers, 1], name='disentangled_latent')
    return


def main():
    # fmap_base = 8192
    # fmap_decay = 1.0
    # fmap_max = 512
    # nf_1 = nf(1, fmap_base, fmap_decay, fmap_max)

    z_dim = 512
    w_dim = 512
    n_mapping = 8
    n_layers = 18
    z = tf.placeholder(tf.float32, shape=[None, 512], name='z')

    generator(z, z_dim, w_dim, n_mapping, n_layers)
    return


if __name__ == '__main__':
    main()
