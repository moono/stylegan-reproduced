import os
import pprint
import numpy as np
import tensorflow as tf
import cv2

from network.generator import generator
from utils.utils import post_process_generator_output


def inference_generator_from_raw_tf(res):
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    index = resolutions.index(res)
    inference_resolutions = resolutions[:index + 1]
    inference_featuremaps = featuremaps[:index + 1]

    # prepare variables & construct generator
    image_out_dir = './assets'
    is_training = False
    z_dim = 512
    g_params = {
        'w_dim': 512,
        'n_mapping': 8,
        'resolutions': inference_resolutions,
        'featuremaps': inference_featuremaps,
        'truncation_psi': 0.7,
        'truncation_cutoff': 8,
    }
    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    alpha = tf.constant(0.0, dtype=tf.float32, shape=[])
    fake_images = generator(z, alpha, g_params, is_training)

    # assign which variables to retore
    var_list = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    pprint.pprint(var_list)

    # restore tools
    model_dir = '/mnt/vision-nas/moono/trained_models/stylegan-reproduced/{:d}x{:d}'.format(res, res)
    model_ckpt = tf.train.latest_checkpoint(os.path.join(model_dir))
    saver = tf.train.Saver(var_list=var_list)

    # set input latent z
    n_output_samples = 4
    rnd = np.random.RandomState(5)
    z_input_np = rnd.randn(n_output_samples, z_dim)

    # generate image with official weights
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_ckpt)

        output_batch = sess.run(fake_images, feed_dict={z: z_input_np})

        for ii in range(n_output_samples):
            output = post_process_generator_output(output_batch[ii, :])
            output = cv2.cvtColor(output, cv2.COLOR_RGB2BGR)

            out_fn = os.path.join(image_out_dir, 'inference-{:d}-{:d}x{:d}.png'.format(ii, res, res))
            cv2.imwrite(out_fn, output)
    return


def main():
    generation_res = 1024
    inference_generator_from_raw_tf(generation_res)
    return


if __name__ == '__main__':
    main()
