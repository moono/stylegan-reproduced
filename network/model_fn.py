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


def preprocess_image(images, res, network_output_res, alpha):
    images = adjust_dynamic_range(images)
    images = random_flip_left_right_nchw(images)
    images = smooth_crossfade(images, alpha)
    images = upscale_to_res(images, res, network_output_res)
    images.set_shape([None, 3, network_output_res, network_output_res])
    return images


def compute_smooth_transition_rate(batch_size, global_step, train_trans_images_per_res):
    train_trans_images_per_res_tensor = tf.constant(train_trans_images_per_res, dtype=tf.float32, shape=[],
                                                    name='train_trans_images_per_res')

    # alpha == 1.0: use only previous resolution output
    # alpha == 0.0: use only current resolution output
    n_cur_img = batch_size * global_step
    n_cur_img = tf.cast(n_cur_img, dtype=tf.float32)

    is_less_op = tf.less_equal(n_cur_img, train_trans_images_per_res_tensor)
    alpha = tf.cond(is_less_op,
                    true_fn=lambda: (train_trans_images_per_res - n_cur_img) / train_trans_images_per_res_tensor,
                    false_fn=lambda: tf.constant(0.0, dtype=tf.float32, shape=[]))

    # alpha = 0.0
    # if n_cur_img < train_trans_images_per_res:
    #     alpha = float(train_trans_images_per_res - n_cur_img) / float(train_trans_images_per_res)
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


def model_fn(features, labels, mode, params):
    # parse inputs
    real_images = features['images']

    # parse params
    z_dim = params['z_dim']
    w_dim = params['w_dim']
    n_mapping = params['n_mapping']
    w_ema_decay = params['w_ema_decay']
    style_mixing_prob = params['style_mixing_prob']
    truncation_psi = params['truncation_psi']
    truncation_cutoff = params['truncation_cutoff']

    res = params['res']
    final_res = params['final_res']
    resolutions = params['resolutions']
    featuremaps = params['featuremaps']
    total_images = params['total_images']
    train_fixed_images_per_res = params['train_fixed_images_per_res']
    train_trans_images_per_res = params['train_trans_images_per_res']
    batch_size = params['batch_size']
    g_learning_rate = params['g_learning_rate']
    d_learning_rate = params['d_learning_rate']

    # w = tf.Variable(0.5, dtype=tf.float32)
    # x = tf.constant(1, dtype=tf.float32)
    # y_ = tf.multiply(x, w)
    # y = tf.constant(1, dtype=tf.float32)
    #
    # actual_images = tf.transpose(real_images, perm=[0, 2, 3, 1])
    # image_shape = tf.shape(actual_images)
    # tf.summary.scalar('height', image_shape[1])
    # tf.summary.scalar('width', image_shape[2])
    # tf.summary.image('images', actual_images[:1])

    # create additional variables & constants
    z = tf.random_normal(shape=[batch_size, z_dim], dtype=tf.float32)
    zero_init = tf.initializers.zeros()
    alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=zero_init, trainable=False)
    w_avg = tf.get_variable('w_avg', shape=[w_dim], dtype=tf.float32, initializer=zero_init, trainable=False)

    # compute current smooth transition alpha
    global_step = tf.train.get_or_create_global_step()
    alpha_const = compute_smooth_transition_rate(batch_size, global_step, train_trans_images_per_res)
    alpha_assign_op = tf.assign(alpha, alpha_const)

    # build network
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    g_params = {
        'alpha': alpha,
        'w_avg': w_avg,
        'z_dim': z_dim,
        'w_dim': w_dim,
        'n_mapping': n_mapping,
        'train_res': res,
        'resolutions': resolutions,
        'featuremaps': featuremaps,
        'w_ema_decay': w_ema_decay,
        'style_mixing_prob': style_mixing_prob,
        'truncation_psi': truncation_psi,
        'truncation_cutoff': truncation_cutoff,
    }

    with tf.control_dependencies([alpha_assign_op]):
        # preprocess input images
        real_images.set_shape([None, 3, res, res])
        real_images = preprocess_image(real_images, res, final_res, alpha=alpha)

        # get generator & discriminator outputs
        fake_images = generator(z, g_params, is_training)
        fake_scores = discriminator(fake_images, alpha, resolutions, featuremaps)
        real_scores = discriminator(real_images, alpha, resolutions, featuremaps)

    # ==================================================================================================================
    # PREDICTION
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={})

    # prepare appropriate training vars
    d_vars, g_vars = filter_trainable_variables(res)

    # compute loss
    loss = tf.reduce_sum(tf.square(tf.subtract(1.0, 0.5)))

    # summaries
    tf.summary.scalar('alpha', alpha)

    # ==================================================================================================================
    # EVALUATION
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops={}, predictions={})

    # ==================================================================================================================
    # TRAINING
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.TRAIN:
        t_var = tf.trainable_variables()
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
        train_op = optimizer.minimize(loss, global_step, t_var)
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops={}, predictions={})


#
# def model_fn(features, labels, mode, params):
#     # parse inputs
#     images = features['images']
#     # n_classes = params['n_classes']
#     is_training = mode == tf.estimator.ModeKeys.TRAIN
#
#     # build network
#     logits = build_network(images, n_classes, is_training)
#
#     # compute predictions
#     predicted_classes = tf.expand_dims(tf.argmax(logits, axis=3, output_type=tf.int32), axis=3)
#     predictions = {
#         'predicted_classes': predicted_classes,
#     }
#
#     # ==================================================================================================================
#     # PREDICTION
#     # ==================================================================================================================
#     if mode == tf.estimator.ModeKeys.PREDICT:
#         return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
#
#     # compute loss
#     t_var = tf.trainable_variables()
#
#     if 'loss_weights' in params:
#         loss_weights = params['loss_weights']
#     else:
#         loss_weights = None
#     weight_decay = 1e-4
#     ce_loss, valid_labels, valid_logits, valid_predic = weighted_loss_fn(n_classes, labels, logits, weight_decay, t_var,
#                                                                          override_weights=loss_weights)
#     dc_loss = dice_loss(n_classes, labels, logits)
#
#     # merge two losses
#     loss = ce_loss + dc_loss
#
#     # compute metrics (metrics used both in training and eval)
#     accuracy = tf.metrics.accuracy(valid_labels, valid_predic, name='pixel_accuracy')
#     mean_iou = tf.metrics.mean_iou(valid_labels, valid_predic, n_classes, name='mean_iou')
#     metrics = {
#         'px_accuracy': accuracy,
#         'mean_iou': mean_iou
#     }
#     # mean_iou not working !?
#     # tf.summary.scalar('mean_iou', mean_iou[1])  # during training
#     tf.summary.scalar('px_accuracy', accuracy[1])  # during training
#
#     # save some images
#     int_label_to_color = params['discrete_cmap']
#     segmentation_labels = decode_labels_tf(labels, int_label_to_color)
#     segmentation_result = decode_labels_tf(predicted_classes, int_label_to_color)
#     tf.summary.image('images', images[:3])
#     tf.summary.image('labels', segmentation_labels[:3])
#     tf.summary.image('result', segmentation_result[:3])
#
#     # ==================================================================================================================
#     # EVALUATION
#     # ==================================================================================================================
#     if mode == tf.estimator.ModeKeys.EVAL:
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics, predictions=predictions)
#
#     # ==================================================================================================================
#     # TRAINING
#     # ==================================================================================================================
#     if mode == tf.estimator.ModeKeys.TRAIN:
#         # more training parameters (adaptive learning rate)
#         learning_rate = params['learning_rate']
#         decay_steps = params['decay_steps']
#         decay_rate = 0.94
#         global_step = tf.train.get_or_create_global_step()
#         learning_rate = tf.train.exponential_decay(learning_rate, global_step, decay_steps, decay_rate, staircase=True)
#         tf.summary.scalar('learning_rate', learning_rate)
#
#         optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
#         with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
#             train_op = optimizer.minimize(loss, global_step, var_list=t_var)
#         return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op, eval_metric_ops=metrics,
#                                           predictions=predictions)
