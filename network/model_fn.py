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


def upscale_to_res(images, res, network_output):
    s = tf.shape(images)
    factor = int(network_output // res)
    # factor = tf.cast(2 ** tf.floor(lod), tf.int32)
    images = tf.reshape(images, [-1, s[1], s[2], 1, s[3], 1])
    images = tf.tile(images, [1, 1, 1, factor, 1, factor])
    images = tf.reshape(images, [-1, s[1], s[2] * factor, s[3] * factor])
    return images


def preprocess_image(images, res, network_output, alpha):
    images = adjust_dynamic_range(images)
    images = random_flip_left_right_nchw(images)
    images = smooth_crossfade(images, alpha)
    images = upscale_to_res(images, res, network_output)
    return images


def model_fn(features, labels, mode, params):
    # parse inputs
    images = features['images']
    # n_classes = params['n_classes']
    is_training = mode == tf.estimator.ModeKeys.TRAIN

    global_step = tf.train.get_or_create_global_step()
    res = params['res']
    final_res = params['final_res']
    resolutions = params['resolutions']
    featuremaps = params['featuremaps']
    total_images_in_each_dataset = params['total_images_in_each_dataset']
    train_fixed_images_per_res = params['train_fixed_images_per_res']
    train_trans_images_per_res = params['train_trans_images_per_res']
    batch_size = params['batch_size']
    g_learning_rate = params['g_learning_rate']
    d_learning_rate = params['d_learning_rate']

    images.set_shape([None, 3, res, res])
    images = preprocess_image(images, res, final_res, alpha=1.0)
    print()

    w = tf.Variable(0.5, dtype=tf.float32)
    x = tf.constant(1, dtype=tf.float32)
    y_ = tf.multiply(x, w)
    y = tf.constant(1, dtype=tf.float32)

    actual_images = tf.transpose(images, perm=[0, 2, 3, 1])
    image_shape = tf.shape(actual_images)
    tf.summary.scalar('height', image_shape[1])
    tf.summary.scalar('width', image_shape[2])
    tf.summary.image('images', actual_images[:1])

    # build network

    # ==================================================================================================================
    # PREDICTION
    # ==================================================================================================================
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions={})

    # ==================================================================================================================
    # EVALUATION
    # ==================================================================================================================
    loss = tf.reduce_sum(tf.square(tf.subtract(y, y_)))
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
