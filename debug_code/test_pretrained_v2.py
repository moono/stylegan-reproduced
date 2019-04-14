import os
import numpy as np
import tensorflow as tf

import cv2


def test_generator_v2():
    from debug_code.stylegan_v2 import generator

    # prepare variables
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    is_training = False
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    # alpha = tf.Variable(initial_value=0.0, trainable=False, name='transition_alpha')
    alpha = tf.constant(0.0, dtype=tf.float32)
    fake_images = generator(z, w_dim, n_mapping, alpha, resolutions, featuremaps, is_training)

    model_dir = '../official-pretrained'
    ckpt_name = 'model.ckpt'
    model_ckpt = os.path.join(model_dir, ckpt_name)

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
        cv2.imwrite('pretrained-out-v2.png', output)
    return


def main():
    test_generator_v2()
    return


if __name__ == '__main__':
    main()
