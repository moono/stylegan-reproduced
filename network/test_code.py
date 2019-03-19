import numpy as np
import tensorflow as tf

from network.stylegan import style_generator, mapping_network, synthesis_network


def test0():
    def nf(stage):
        fmap_base = 8192
        fmap_decay = 1.0
        fmap_max = 512
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    for res in range(2, 11):
        out_nf = nf(res - 1)
        print('res: {}, nf: {}'.format(res, out_nf))
    return


def test1():
    batch_size = 1
    num_layers = 18
    style_mixing_prob = 0.9

    lod_in = tf.get_variable('lod', initializer=np.float32(0), trainable=False)
    # latents_in = tf.constant(1.0, dtype=tf.float32, shape=[batch_size, 512])
    dlatents1 = tf.constant(0.0, dtype=tf.float32, shape=[batch_size, num_layers, 512])
    dlatents2 = tf.constant(0.5, dtype=tf.float32, shape=[batch_size, num_layers, 512])

    # latents2 = tf.random_normal(tf.shape(latents_in))
    # dlatents2 = components.mapping.get_output_for(latents2, labels_in, **kwargs)
    layer_idx = np.arange(num_layers)[np.newaxis, :, np.newaxis]
    cur_layers = num_layers - tf.cast(lod_in, tf.int32) * 2
    mixing_cutoff = tf.cond(
        tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
        lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
        lambda: cur_layers)
    dlatents = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(dlatents1)), dlatents1, dlatents2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        out_cur_layers, out_mixing_cutoff, out_dlatents = sess.run([cur_layers, mixing_cutoff, dlatents])
        print(out_cur_layers)
        print(out_mixing_cutoff)
        # print(out_dlatents)
        print(np.squeeze(out_dlatents, axis=0))
        print()
    return


def test2():
    # prepare generator variables
    batch_size = 32
    z_dim = 512
    w_dim = 512
    random_noise = True
    n_mapping = 8
    final_output_resolution = 1024
    resolution_log2 = int(np.log2(final_output_resolution))
    resolutions = [2 ** (power + 1) for power in range(1, resolution_log2)]

    z = tf.random_normal(shape=[batch_size, z_dim], dtype=tf.float32, name='latent_z')
    # z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    w = mapping_network(z, w_dim, n_mapping)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        w_out = sess.run(w)

        # w_out = sess.run(w, feed_dict={
        #     z: np.random.normal(size=[batch_size, z_dim])
        # })

        print(w_out)
    return


def main():
    test0()
    # test1()
    # test2()
    return


if __name__ == '__main__':
    main()
