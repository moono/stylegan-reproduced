import os
import numpy as np
import tensorflow as tf
import cv2

from network.stylegan_same import generator


official_code_g_synthesis_t_vars = [
    'G_synthesis/4x4/Const/const:0',
    'G_synthesis/4x4/Const/Noise/weight:0',
    'G_synthesis/4x4/Const/bias:0',
    'G_synthesis/4x4/Const/StyleMod/weight:0',
    'G_synthesis/4x4/Const/StyleMod/bias:0',
    'G_synthesis/4x4/Conv/weight:0',
    'G_synthesis/4x4/Conv/Noise/weight:0',
    'G_synthesis/4x4/Conv/bias:0',
    'G_synthesis/4x4/Conv/StyleMod/weight:0',
    'G_synthesis/4x4/Conv/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod8/weight:0',
    'G_synthesis/ToRGB_lod8/bias:0',
    'G_synthesis/8x8/Conv0_up/weight:0',
    'G_synthesis/8x8/Conv0_up/Noise/weight:0',
    'G_synthesis/8x8/Conv0_up/bias:0',
    'G_synthesis/8x8/Conv0_up/StyleMod/weight:0',
    'G_synthesis/8x8/Conv0_up/StyleMod/bias:0',
    'G_synthesis/8x8/Conv1/weight:0',
    'G_synthesis/8x8/Conv1/Noise/weight:0',
    'G_synthesis/8x8/Conv1/bias:0',
    'G_synthesis/8x8/Conv1/StyleMod/weight:0',
    'G_synthesis/8x8/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod7/weight:0',
    'G_synthesis/ToRGB_lod7/bias:0',
    'G_synthesis/16x16/Conv0_up/weight:0',
    'G_synthesis/16x16/Conv0_up/Noise/weight:0',
    'G_synthesis/16x16/Conv0_up/bias:0',
    'G_synthesis/16x16/Conv0_up/StyleMod/weight:0',
    'G_synthesis/16x16/Conv0_up/StyleMod/bias:0',
    'G_synthesis/16x16/Conv1/weight:0',
    'G_synthesis/16x16/Conv1/Noise/weight:0',
    'G_synthesis/16x16/Conv1/bias:0',
    'G_synthesis/16x16/Conv1/StyleMod/weight:0',
    'G_synthesis/16x16/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod6/weight:0',
    'G_synthesis/ToRGB_lod6/bias:0',
    'G_synthesis/32x32/Conv0_up/weight:0',
    'G_synthesis/32x32/Conv0_up/Noise/weight:0',
    'G_synthesis/32x32/Conv0_up/bias:0',
    'G_synthesis/32x32/Conv0_up/StyleMod/weight:0',
    'G_synthesis/32x32/Conv0_up/StyleMod/bias:0',
    'G_synthesis/32x32/Conv1/weight:0',
    'G_synthesis/32x32/Conv1/Noise/weight:0',
    'G_synthesis/32x32/Conv1/bias:0',
    'G_synthesis/32x32/Conv1/StyleMod/weight:0',
    'G_synthesis/32x32/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod5/weight:0',
    'G_synthesis/ToRGB_lod5/bias:0',
    'G_synthesis/64x64/Conv0_up/weight:0',
    'G_synthesis/64x64/Conv0_up/Noise/weight:0',
    'G_synthesis/64x64/Conv0_up/bias:0',
    'G_synthesis/64x64/Conv0_up/StyleMod/weight:0',
    'G_synthesis/64x64/Conv0_up/StyleMod/bias:0',
    'G_synthesis/64x64/Conv1/weight:0',
    'G_synthesis/64x64/Conv1/Noise/weight:0',
    'G_synthesis/64x64/Conv1/bias:0',
    'G_synthesis/64x64/Conv1/StyleMod/weight:0',
    'G_synthesis/64x64/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod4/weight:0',
    'G_synthesis/ToRGB_lod4/bias:0',
    'G_synthesis/128x128/Conv0_up/weight:0',
    'G_synthesis/128x128/Conv0_up/Noise/weight:0',
    'G_synthesis/128x128/Conv0_up/bias:0',
    'G_synthesis/128x128/Conv0_up/StyleMod/weight:0',
    'G_synthesis/128x128/Conv0_up/StyleMod/bias:0',
    'G_synthesis/128x128/Conv1/weight:0',
    'G_synthesis/128x128/Conv1/Noise/weight:0',
    'G_synthesis/128x128/Conv1/bias:0',
    'G_synthesis/128x128/Conv1/StyleMod/weight:0',
    'G_synthesis/128x128/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod3/weight:0',
    'G_synthesis/ToRGB_lod3/bias:0',
    'G_synthesis/256x256/Conv0_up/weight:0',
    'G_synthesis/256x256/Conv0_up/Noise/weight:0',
    'G_synthesis/256x256/Conv0_up/bias:0',
    'G_synthesis/256x256/Conv0_up/StyleMod/weight:0',
    'G_synthesis/256x256/Conv0_up/StyleMod/bias:0',
    'G_synthesis/256x256/Conv1/weight:0',
    'G_synthesis/256x256/Conv1/Noise/weight:0',
    'G_synthesis/256x256/Conv1/bias:0',
    'G_synthesis/256x256/Conv1/StyleMod/weight:0',
    'G_synthesis/256x256/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod2/weight:0',
    'G_synthesis/ToRGB_lod2/bias:0',
    'G_synthesis/512x512/Conv0_up/weight:0',
    'G_synthesis/512x512/Conv0_up/Noise/weight:0',
    'G_synthesis/512x512/Conv0_up/bias:0',
    'G_synthesis/512x512/Conv0_up/StyleMod/weight:0',
    'G_synthesis/512x512/Conv0_up/StyleMod/bias:0',
    'G_synthesis/512x512/Conv1/weight:0',
    'G_synthesis/512x512/Conv1/Noise/weight:0',
    'G_synthesis/512x512/Conv1/bias:0',
    'G_synthesis/512x512/Conv1/StyleMod/weight:0',
    'G_synthesis/512x512/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod1/weight:0',
    'G_synthesis/ToRGB_lod1/bias:0',
    'G_synthesis/1024x1024/Conv0_up/weight:0',
    'G_synthesis/1024x1024/Conv0_up/Noise/weight:0',
    'G_synthesis/1024x1024/Conv0_up/bias:0',
    'G_synthesis/1024x1024/Conv0_up/StyleMod/weight:0',
    'G_synthesis/1024x1024/Conv0_up/StyleMod/bias:0',
    'G_synthesis/1024x1024/Conv1/weight:0',
    'G_synthesis/1024x1024/Conv1/Noise/weight:0',
    'G_synthesis/1024x1024/Conv1/bias:0',
    'G_synthesis/1024x1024/Conv1/StyleMod/weight:0',
    'G_synthesis/1024x1024/Conv1/StyleMod/bias:0',
    'G_synthesis/ToRGB_lod0/weight:0',
    'G_synthesis/ToRGB_lod0/bias:0',
]

official_code_g_mapping_t_vars = [
    'G_mapping/Dense0/weight:0',
    'G_mapping/Dense0/bias:0',
    'G_mapping/Dense1/weight:0',
    'G_mapping/Dense1/bias:0',
    'G_mapping/Dense2/weight:0',
    'G_mapping/Dense2/bias:0',
    'G_mapping/Dense3/weight:0',
    'G_mapping/Dense3/bias:0',
    'G_mapping/Dense4/weight:0',
    'G_mapping/Dense4/bias:0',
    'G_mapping/Dense5/weight:0',
    'G_mapping/Dense5/bias:0',
    'G_mapping/Dense6/weight:0',
    'G_mapping/Dense6/bias:0',
    'G_mapping/Dense7/weight:0',
    'G_mapping/Dense7/bias:0'
]


def test_generator():
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

    t_vars = tf.trainable_variables()
    var_list = dict()
    for official in official_code_g_synthesis_t_vars:
        temp = official.split('/', 1)
        mine = '{}/{}'.format(temp[0].lower(), temp[1])
        skimmed_official = official.split(':')[0]

        for v in t_vars:
            if v.name == mine:
                var_list[skimmed_official] = v
                break

    for official in official_code_g_mapping_t_vars:
        temp = official.split('/', 1)
        mine = '{}/{}'.format(temp[0].lower(), temp[1])
        skimmed_official = official.split(':')[0]

        for v in t_vars:
            if v.name == mine:
                var_list[skimmed_official] = v
                break

    saver = tf.train.Saver(var_list=var_list)

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
        cv2.imwrite('pretrained-out.png', output)
    return


def main():
    test_generator()
    return


if __name__ == '__main__':
    main()
