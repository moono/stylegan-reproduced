import os
import numpy as np
import tensorflow as tf
import cv2


def test_generator_official():
    from debug_code.stylegan_official import G_style

    # prepare generator variables
    is_training = False
    z_dim = 512

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    fake_images = G_style(z, is_training)

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
        cv2.imwrite('pretrained-out-official.png', output)
    return


def main():
    test_generator_official()
    return


if __name__ == '__main__':
    main()
