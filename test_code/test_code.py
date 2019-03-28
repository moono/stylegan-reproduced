import numpy as np
import tensorflow as tf


def test0():
    def nf(stage):
        fmap_base = 8192
        fmap_decay = 1.0
        fmap_max = 512
        return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)

    for res in range(1, 11):
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
    from network.stylegan_same import g_mapping

    # prepare generator variables
    batch_size = 32
    z_dim = 512
    w_dim = 512
    random_noise = True
    n_mapping = 8
    final_output_resolution = 1024
    resolution_log2 = int(np.log2(final_output_resolution))
    resolutions = [2 ** (power + 1) for power in range(1, resolution_log2)]
    n_broadcast = len(resolutions) * 2

    z = tf.random_normal(shape=[batch_size, z_dim], dtype=tf.float32, name='latent_z')
    # z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    w = g_mapping(z, w_dim, n_mapping, n_broadcast)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        w_out = sess.run(w)

        # w_out = sess.run(w, feed_dict={
        #     z: np.random.normal(size=[batch_size, z_dim])
        # })

        print(w_out)
    return


def test3():
    import pprint
    import pickle

    def training_schedule(cur_nimg):
        lod_training_kimg = 600
        lod_transition_kimg = 600
        resolution_log2 = 10
        lod_initial_resolution = 8
        minibatch_base = 2
        minibatch_dict = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4, 512: 2}
        num_gpus = 1
        max_minibatch_per_gpu = dict()

        lrate_rampup_kimg = 0
        G_lrate_base = 0.001
        D_lrate_base = 0.001
        G_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
        D_lrate_dict = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

        s = dict()
        s['kimg'] = cur_nimg / 1000.0

        # Training phase.
        phase_dur = lod_training_kimg + lod_transition_kimg
        phase_idx = int(np.floor(s['kimg'] / phase_dur)) if phase_dur > 0 else 0
        phase_kimg = s['kimg'] - phase_idx * phase_dur

        # Level-of-detail and resolution.
        s['lod'] = resolution_log2
        s['lod'] -= np.floor(np.log2(lod_initial_resolution))
        s['lod'] -= phase_idx
        if lod_transition_kimg > 0:
            s['lod'] -= max(phase_kimg - lod_training_kimg, 0.0) / lod_transition_kimg
        s['lod'] = max(s['lod'], 0.0)
        s['resolution'] = 2 ** (resolution_log2 - int(np.floor(s['lod'])))

        # Minibatch size.
        s['minibatch'] = minibatch_dict.get(s['resolution'], minibatch_base)
        s['minibatch'] -= s['minibatch'] % num_gpus
        if s['resolution'] in max_minibatch_per_gpu:
            s['minibatch'] = min(s['minibatch'], max_minibatch_per_gpu[s['resolution']] * num_gpus)

        # Learning rate.
        s['G_lrate'] = G_lrate_dict.get(s['resolution'], G_lrate_base)
        s['D_lrate'] = D_lrate_dict.get(s['resolution'], D_lrate_base)
        if lrate_rampup_kimg > 0:
            rampup = min(s['kimg'] / lrate_rampup_kimg, 1.0)
            s['G_lrate'] *= rampup
            s['D_lrate'] *= rampup

        # # Other parameters.
        # s.tick_kimg = tick_kimg_dict.get(s['resolution'], tick_kimg_base)
        return s

    # diff = 0.00010666666666647728

    cur_img = 0
    sched = training_schedule(cur_img)
    lod = sched['lod']
    pprint.pprint(sched)

    resolution = list()
    lod_list = list()
    epochs = 1000
    while cur_img < 70000 * epochs:
        cur_img += sched['minibatch']
        sched = training_schedule(cur_img)
        if lod != sched['lod']:
            lod_list.append(sched['lod'])
            resolution.append(sched['resolution'])

            lod = sched['lod']

    with open('schedule_ex_lod.pkl', 'wb') as f:
        pickle.dump(lod_list, f)
    with open('schedule_ex_res.pkl', 'wb') as f:
        pickle.dump(resolution, f)

    return


def test4():
    import pickle
    import matplotlib.pyplot as plt

    with open('schedule_ex_lod.pkl', 'rb') as f:
        lod_list = pickle.load(f)
    with open('schedule_ex_res.pkl', 'rb') as f:
        resolution = pickle.load(f)

    unique_res, unique_res_indices = np.unique(np.array(resolution), return_index=True)
    fig, ax = plt.subplots()
    plt.plot(lod_list, '--b', label='lod')
    plt.legend()
    ax.tick_params('vals', colors='b')

    # Get second axis
    ax2 = ax.twinx()
    plt.plot(resolution, '--r', label='res')
    plt.legend()
    # ax.tick_params('vals', colors='r')
    ax2.set_yticks(unique_res)

    plt.xticks(unique_res_indices)
    # plt.grid(True, which='both')
    plt.gca().xaxis.grid(True)
    plt.gca().yaxis.grid(True)
    plt.show()
    return


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


def test5():
    w = tf.Variable(0, dtype=tf.float32)
    w_broadcasted = w_broadcaster(w, 1, 2)
    ema = tf.train.ExponentialMovingAverage(decay=0.9)
    m = ema.apply([w])
    av = ema.average(w)

    x = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.float32, [None])
    y_ = tf.multiply(x, w)

    with tf.control_dependencies([m]):
        # w_broadcasted[0] = tf.identity(w_broadcasted[0])
        loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
        train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            _, w_, av_, w0, w1 = sess.run([train, w, av, w_broadcasted[0], w_broadcasted[1]], feed_dict={x: [1], y: [10]})

            if i % 10 == 0:
                print(i, ': ', w_, ',', av_, ',', w0, ',', w1)
    return


def test6():
    w = tf.Variable(0, dtype=tf.float32)

    x = tf.placeholder(tf.float32, [None])
    y = tf.placeholder(tf.float32, [None])
    y_ = tf.multiply(x, w)

    loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.1)
    train = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(100):
            _, w_ = sess.run([train, w], feed_dict={x: [1], y: [10]})

            if i % 10 == 0:
                print(i, ': ', w_)
    return


def test7():
    def create_variable():
        with tf.variable_scope("foo", reuse=tf.AUTO_REUSE):
            with tf.variable_scope("bar"):
                v = tf.get_variable("v", [1])
        return v

    v1 = create_variable()
    v2 = create_variable()
    return


def test8():
    from network.common_ops import lerp

    w_dim = 5
    n_broadcast = 18
    truncation_psi = 0.7
    truncation_cutoff = 8
    w_broadcasted = tf.constant(1.0, dtype=tf.float32, shape=[1, n_broadcast, w_dim])
    w_avg = tf.constant(0.5, dtype=tf.float32, shape=[w_dim])

    layer_idx = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
    ones = np.ones(layer_idx.shape, dtype=np.float32)
    coefs = tf.where(layer_idx < truncation_cutoff, truncation_psi * ones, ones)
    w_broadcasted = lerp(w_avg, w_broadcasted, coefs)

    print('layer_idx: {}'.format(layer_idx))
    print('ones: {}'.format(ones))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        coefs_out, out = sess.run([coefs, w_broadcasted])
        print(coefs_out)
        print(out)
    return


def test9():
    style_mixing_prob = 0.9
    n_broadcast = 18
    cur_layers = 7

    w_dim = 5
    lod_in = tf.constant(1.0, dtype=tf.float32)
    w_broadcasted1 = tf.constant(1.0, dtype=tf.float32, shape=[1, n_broadcast, w_dim])
    w_broadcasted2 = tf.constant(0.5, dtype=tf.float32, shape=[1, n_broadcast, w_dim])

    layer_idx = np.arange(n_broadcast)[np.newaxis, :, np.newaxis]
    mixing_cutoff = tf.cond(
        tf.random_uniform([], 0.0, 1.0) < style_mixing_prob,
        lambda: tf.random_uniform([], 1, cur_layers, dtype=tf.int32),
        lambda: cur_layers)
    w_broadcasted1 = tf.where(tf.broadcast_to(layer_idx < mixing_cutoff, tf.shape(w_broadcasted1)), w_broadcasted1, w_broadcasted2)

    print('layer_idx: {}'.format(layer_idx))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        mixing_cutoffout, out = sess.run([mixing_cutoff, w_broadcasted1])
        print(mixing_cutoffout)
        print(out)
    return


def main():
    test0()
    # test1()
    # test2()
    # test3()
    # test4()
    # test5()
    # test6()
    # test7()
    # test8()
    # test9()
    return


if __name__ == '__main__':
    main()
