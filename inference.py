import os
import pprint
import numpy as np
import tensorflow as tf
import cv2

from network.generator import generator


def inference(infer_res):
    # prepare variables
    zero_init = tf.initializers.zeros()

    is_training = False
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    # train_res = 32
    # alpha = tf.get_variable('alpha', shape=[], dtype=tf.float32, initializer=zero_init, trainable=False)
    alpha = tf.constant(0.0, dtype=tf.float32)
    w_avg = tf.get_variable('w_avg', shape=[w_dim], dtype=tf.float32, initializer=zero_init, trainable=False)
    w_ema_decay = 0.995
    style_mixing_prob = 0.9
    truncation_psi = 0.7
    truncation_cutoff = 8

    # new resolutions & featuremaps
    inference_res_index = resolutions.index(infer_res)
    inference_resolutions = resolutions[:inference_res_index + 1]
    inference_featuremaps = featuremaps[:inference_res_index + 1]

    g_params = {
        'alpha': alpha,
        'w_avg': w_avg,
        'z_dim': z_dim,
        'w_dim': w_dim,
        'n_mapping': n_mapping,
        # 'train_res': train_res,
        'resolutions': inference_resolutions,
        'featuremaps': inference_featuremaps,
        'w_ema_decay': w_ema_decay,
        'style_mixing_prob': style_mixing_prob,
        'truncation_psi': truncation_psi,
        'truncation_cutoff': truncation_cutoff,
    }

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    fake_images = generator(z, g_params, is_training)
    print('output fake image shape: {}'.format(fake_images.shape))

    all_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    pprint.pprint(all_vars)
    # for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES):
    #     if not v.trainable:

    model_save_base_dir = '/mnt/vision-nas/moono/trained_models/stylegan-reproduced/{}x{}'.format(infer_res, infer_res)
    model_ckpt = tf.train.latest_checkpoint(model_save_base_dir)

    saver = tf.train.Saver()

    rnd = np.random.RandomState(5)
    z_input_np = rnd.randn(1, z_dim)
    drange_min, drange_max = -1.0, 1.0
    scale = 255.0 / (drange_max - drange_min)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_ckpt)

        output = sess.run(fake_images, feed_dict={z: z_input_np})
        print(output.shape)

        output = np.squeeze(output, axis=0)
        output = np.transpose(output, axes=[1, 2, 0])
        output = output * scale + (0.5 - drange_min * scale)
        output = np.clip(output, 0, 255)
        output = output.astype('uint8')
        output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)
        cv2.imwrite('inference-out-{}x{}.png'.format(infer_res, infer_res), output)
    return


def main():
    infer_res = 64
    inference(infer_res)
    return


if __name__ == '__main__':
    main()
