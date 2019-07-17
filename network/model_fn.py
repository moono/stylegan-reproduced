import numpy as np
import tensorflow as tf

from network.common_ops import lerp
from network.generator import generator
from network.discriminator import discriminator


def adjust_dynamic_range(images):
    drange_in = [0.0, 255.0]
    drange_out = [-1.0, 1.0]
    scale = (drange_out[1] - drange_out[0]) / (drange_in[1] - drange_in[0])
    bias = drange_out[0] - drange_in[0] * scale
    images = images * scale + bias
    return images


def random_flip_left_right_nchw(images):
    s = tf.shape(images)
    mask = tf.random_uniform([s[0], 1, 1, 1], 0.0, 1.0)
    mask = tf.tile(mask, [1, s[1], s[2], s[3]])
    images = tf.where(mask < 0.5, images, tf.reverse(images, axis=[3]))
    return images


def smooth_crossfade(images, alpha):
    s = tf.shape(images)
    y = tf.reshape(images, [-1, s[1], s[2] // 2, 2, s[3] // 2, 2])
    y = tf.reduce_mean(y, axis=[3, 5], keepdims=True)
    y = tf.tile(y, [1, 1, 1, 2, 1, 2])
    y = tf.reshape(y, [-1, s[1], s[2], s[3]])
    images = lerp(images, y, alpha)
    return images


def upscale_to_res(images, res, network_output_res):
    s = tf.shape(images)
    factor = int(network_output_res // res)
    # factor = tf.cast(2 ** tf.floor(lod), tf.int32)
    images = tf.reshape(images, [-1, s[1], s[2], 1, s[3], 1])
    images = tf.tile(images, [1, 1, 1, factor, 1, factor])
    images = tf.reshape(images, [-1, s[1], s[2] * factor, s[3] * factor])
    return images


def preprocess_fit_train_image(images, res, alpha):
    images = adjust_dynamic_range(images)
    images = random_flip_left_right_nchw(images)
    images = smooth_crossfade(images, alpha)
    images.set_shape([None, 3, res, res])
    return images


def smooth_transition_state(batch_size, global_step, train_trans_images_per_res_tensor, zero_constant):
    # alpha == 1.0: use only previous resolution output
    # alpha == 0.0: use only current resolution output
    n_cur_img = batch_size * global_step
    n_cur_img = tf.cast(n_cur_img, dtype=tf.float32)

    do_transition_train = tf.math.greater(train_trans_images_per_res_tensor, 0)
    is_transition_state = tf.math.logical_and(do_transition_train,
                                              tf.less_equal(n_cur_img, train_trans_images_per_res_tensor))
    alpha = tf.cond(is_transition_state,
                    true_fn=lambda: (train_trans_images_per_res_tensor - n_cur_img) / train_trans_images_per_res_tensor,
                    false_fn=lambda: zero_constant)
    return alpha


def filter_trainable_variables(res):
    res_in_focus = [2 ** r for r in range(int(np.log2(res)), 1, -1)]
    res_in_focus = res_in_focus[::-1]

    t_vars = tf.trainable_variables()
    d_vars = list()
    g_vars = list()
    for var in t_vars:
        if var.name.startswith('g_mapping'):
            g_vars.append(var)
        elif var.name.startswith('g_synthesis'):
            for r in res_in_focus:
                if '{:d}x{:d}'.format(r, r) in var.name:
                    g_vars.append(var)
        elif var.name.startswith('discriminator'):
            for r in res_in_focus:
                if '{:d}x{:d}'.format(r, r) in var.name:
                    d_vars.append(var)

    return d_vars, g_vars


def compute_loss(real_images, real_scores, fake_scores):
    r1_gamma, r2_gamma = 10.0, 0.0

    # discriminator loss: gradient penalty
    d_loss_gan = tf.nn.softplus(fake_scores) + tf.nn.softplus(-real_scores)
    real_loss = tf.reduce_sum(real_scores)
    real_grads = tf.gradients(real_loss, [real_images])[0]
    r1_penalty = tf.reduce_sum(tf.square(real_grads), axis=[1, 2, 3])
    # r1_penalty = tf.reduce_mean(r1_penalty)
    d_loss = d_loss_gan + r1_penalty * (r1_gamma * 0.5)
    d_loss = tf.reduce_mean(d_loss)

    # generator loss: logistic nonsaturating
    g_loss = tf.nn.softplus(-fake_scores)
    g_loss = tf.reduce_mean(g_loss)
    return d_loss, g_loss, tf.reduce_mean(d_loss_gan), tf.reduce_mean(r1_penalty)


def convert_to_rgb_images(images):
    drange_min, drange_max = -1.0, 1.0
    scale = 255.0 / (drange_max - drange_min)

    output = tf.transpose(images, perm=[0, 2, 3, 1])
    output = output * scale + (0.5 - drange_min * scale)
    output = tf.clip_by_value(output, 0.0, 255.0)
    output = tf.cast(output, dtype=tf.uint8)
    return output


def model_fn(features, labels, mode, params):
    # parse params
    w_dim = params['w_dim']
    n_mapping = params['n_mapping']
    resolutions = params['resolutions']
    featuremaps = params['featuremaps']
    style_mixing_prob = params['style_mixing_prob']
    truncation_psi = params['truncation_psi']
    truncation_cutoff = params['truncation_cutoff']
    do_train_trans = params['do_train_trans']
    train_trans_images_per_res = params['train_trans_images_per_res']
    batch_size = params['batch_size']

    # additional params
    train_res = resolutions[-1]
    w_ema_decay = params['w_ema_decay']
    is_training = mode == tf.estimator.ModeKeys.TRAIN
    global_step = tf.train.get_or_create_global_step()

    # set generator & discriminator parameters
    g_params = {
        'w_dim': w_dim,
        'n_mapping': n_mapping,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
        'w_ema_decay': w_ema_decay,
        'style_mixing_prob': style_mixing_prob,
        'truncation_psi': truncation_psi,
        'truncation_cutoff': truncation_cutoff,
    }
    d_params = {
        'resolutions': resolutions,
        'featuremaps': featuremaps,
    }

    # additional variables (reuse zero constants)
    zero_constant = tf.constant(0.0, dtype=tf.float32, shape=[])

    # additional variables (for training only)
    train_trans_images_per_res_tensor = tf.constant(train_trans_images_per_res, dtype=tf.float32, shape=[],
                                                    name='train_trans_images_per_res')

    # smooth transition variable
    alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32,
                            initializer=tf.initializers.ones() if do_train_trans else tf.initializers.zeros(),
                            trainable=False, aggregation=tf.VariableAggregation.ONLY_FIRST_TOWER)

    # determine smooth transition state and compute alpha value
    alpha_const = smooth_transition_state(batch_size, global_step, train_trans_images_per_res_tensor, zero_constant)
    alpha_assign_op = tf.assign(alpha, alpha_const)

    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.TRAIN:
        # get training specific parameters
        z_dim = params['z_dim']
        g_learning_rate = params['g_learning_rate']
        d_learning_rate = params['d_learning_rate']

        # get inputs: latent z, real image input
        z = tf.random_normal(shape=[batch_size, z_dim], dtype=tf.float32)
        real_images = features['real_images']

        # get network outputs
        with tf.control_dependencies([alpha_assign_op]):
            # preprocess input images
            real_images.set_shape([None, 3, train_res, train_res])
            real_images = preprocess_fit_train_image(real_images, train_res, alpha=alpha)

            # create generator output
            fake_images = generator(z, alpha, g_params, is_training=True)

            # get discriminator outputs
            fake_scores = discriminator(fake_images, alpha, d_params)
            real_scores = discriminator(real_images, alpha, d_params)

        # prepare appropriate training vars
        d_vars, g_vars = filter_trainable_variables(train_res)

        # compute loss
        d_loss, g_loss, d_loss_gan, r1_penalty = compute_loss(real_images, real_scores, fake_scores)

        # combine loss for tf.estimator architecture
        loss = d_loss + g_loss

        # prepare optimizer & training ops
        d_optimizer = tf.train.AdamOptimizer(g_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        g_optimizer = tf.train.AdamOptimizer(d_learning_rate, beta1=0.0, beta2=0.99, epsilon=1e-8)
        d_train_opt = d_optimizer.minimize(d_loss, var_list=d_vars)
        g_train_opt = g_optimizer.minimize(g_loss, var_list=g_vars, global_step=global_step)
        train_op = tf.group(d_train_opt, g_train_opt)

        # add summaries
        fake_images_eval = generator(z, zero_constant, g_params, is_training=False)
        summary_real_images = convert_to_rgb_images(real_images)
        summary_fake_images = convert_to_rgb_images(fake_images)
        summary_fake_images_eval = convert_to_rgb_images(fake_images_eval)
        tf.summary.scalar('alpha', alpha)
        tf.summary.scalar('d_loss_gan', d_loss_gan)
        tf.summary.scalar('r1_penalty', r1_penalty)
        tf.summary.scalar('d_loss', d_loss)
        tf.summary.scalar('g_loss', g_loss)
        tf.summary.image('real_images', summary_real_images[:5], max_outputs=5)
        tf.summary.image('fake_images', summary_fake_images[:5], max_outputs=5)
        tf.summary.image('fake_images_eval', summary_fake_images_eval[:5], max_outputs=5)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops={}, predictions={})

    # ==================================================================================================================
    # EVALUATION
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.EVAL:
        # tf.summary.image not working on eval mode?
        return tf.estimator.EstimatorSpec(mode=mode, loss=zero_constant, eval_metric_ops={})

    # ==================================================================================================================
    # PREDICTION
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.PREDICT:
        # get input latent z
        z = features['z']

        # create generator output for evalutation & prediction
        fake_images_eval = generator(z, zero_constant, g_params, is_training=False)

        predictions = {
            'fake_images': fake_images_eval
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
