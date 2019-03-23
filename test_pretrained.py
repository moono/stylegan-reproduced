import os
import numpy as np
import tensorflow as tf
from network.stylegan import generator


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

my_g_synthesis = [
    'generator/synthesis/4x4/const/const:0',
    'generator/synthesis/4x4/const/add_noise/weight:0',
    'generator/synthesis/4x4/const/add_noise/bias:0',
    'generator/synthesis/4x4/const/style_mod/equalized_dense/weight:0',
    'generator/synthesis/4x4/const/style_mod/equalized_dense/bias:0',
    'generator/synthesis/4x4/conv/equalized_conv2d/weight:0',
    'generator/synthesis/4x4/conv/add_noise/weight:0',
    'generator/synthesis/4x4/conv/add_noise/bias:0',
    'generator/synthesis/4x4/conv/style_mod/equalized_dense/weight:0',
    'generator/synthesis/4x4/conv/style_mod/equalized_dense/bias:0',
    'generator/synthesis/4x4/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/4x4/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/8x8/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/8x8/conv0/add_noise/weight:0',
    'generator/synthesis/8x8/conv0/add_noise/bias:0',
    'generator/synthesis/8x8/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/8x8/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/8x8/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/8x8/conv1/add_noise/weight:0',
    'generator/synthesis/8x8/conv1/add_noise/bias:0',
    'generator/synthesis/8x8/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/8x8/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/8x8/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/8x8/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/16x16/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/16x16/conv0/add_noise/weight:0',
    'generator/synthesis/16x16/conv0/add_noise/bias:0',
    'generator/synthesis/16x16/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/16x16/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/16x16/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/16x16/conv1/add_noise/weight:0',
    'generator/synthesis/16x16/conv1/add_noise/bias:0',
    'generator/synthesis/16x16/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/16x16/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/16x16/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/16x16/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/32x32/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/32x32/conv0/add_noise/weight:0',
    'generator/synthesis/32x32/conv0/add_noise/bias:0',
    'generator/synthesis/32x32/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/32x32/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/32x32/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/32x32/conv1/add_noise/weight:0',
    'generator/synthesis/32x32/conv1/add_noise/bias:0',
    'generator/synthesis/32x32/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/32x32/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/32x32/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/32x32/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/64x64/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/64x64/conv0/add_noise/weight:0',
    'generator/synthesis/64x64/conv0/add_noise/bias:0',
    'generator/synthesis/64x64/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/64x64/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/64x64/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/64x64/conv1/add_noise/weight:0',
    'generator/synthesis/64x64/conv1/add_noise/bias:0',
    'generator/synthesis/64x64/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/64x64/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/64x64/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/64x64/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/128x128/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/128x128/conv0/add_noise/weight:0',
    'generator/synthesis/128x128/conv0/add_noise/bias:0',
    'generator/synthesis/128x128/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/128x128/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/128x128/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/128x128/conv1/add_noise/weight:0',
    'generator/synthesis/128x128/conv1/add_noise/bias:0',
    'generator/synthesis/128x128/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/128x128/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/128x128/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/128x128/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/256x256/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/256x256/conv0/add_noise/weight:0',
    'generator/synthesis/256x256/conv0/add_noise/bias:0',
    'generator/synthesis/256x256/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/256x256/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/256x256/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/256x256/conv1/add_noise/weight:0',
    'generator/synthesis/256x256/conv1/add_noise/bias:0',
    'generator/synthesis/256x256/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/256x256/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/256x256/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/256x256/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/512x512/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/512x512/conv0/add_noise/weight:0',
    'generator/synthesis/512x512/conv0/add_noise/bias:0',
    'generator/synthesis/512x512/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/512x512/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/512x512/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/512x512/conv1/add_noise/weight:0',
    'generator/synthesis/512x512/conv1/add_noise/bias:0',
    'generator/synthesis/512x512/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/512x512/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/512x512/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/512x512/to_rgb/equalized_conv2d/bias:0',
    'generator/synthesis/1024x1024/conv0/equalized_conv2d/weight:0',
    'generator/synthesis/1024x1024/conv0/add_noise/weight:0',
    'generator/synthesis/1024x1024/conv0/add_noise/bias:0',
    'generator/synthesis/1024x1024/conv0/style_mod/equalized_dense/weight:0',
    'generator/synthesis/1024x1024/conv0/style_mod/equalized_dense/bias:0',
    'generator/synthesis/1024x1024/conv1/equalized_conv2d/weight:0',
    'generator/synthesis/1024x1024/conv1/add_noise/weight:0',
    'generator/synthesis/1024x1024/conv1/add_noise/bias:0',
    'generator/synthesis/1024x1024/conv1/style_mod/equalized_dense/weight:0',
    'generator/synthesis/1024x1024/conv1/style_mod/equalized_dense/bias:0',
    'generator/synthesis/1024x1024/to_rgb/equalized_conv2d/weight:0',
    'generator/synthesis/1024x1024/to_rgb/equalized_conv2d/bias:0',
]

my_g_mapping = [
    'generator/mapping/layer_0/equalized_dense/weight:0',
    'generator/mapping/layer_0/equalized_dense/bias:0',
    'generator/mapping/layer_1/equalized_dense/weight:0',
    'generator/mapping/layer_1/equalized_dense/bias:0',
    'generator/mapping/layer_2/equalized_dense/weight:0',
    'generator/mapping/layer_2/equalized_dense/bias:0',
    'generator/mapping/layer_3/equalized_dense/weight:0',
    'generator/mapping/layer_3/equalized_dense/bias:0',
    'generator/mapping/layer_4/equalized_dense/weight:0',
    'generator/mapping/layer_4/equalized_dense/bias:0',
    'generator/mapping/layer_5/equalized_dense/weight:0',
    'generator/mapping/layer_5/equalized_dense/bias:0',
    'generator/mapping/layer_6/equalized_dense/weight:0',
    'generator/mapping/layer_6/equalized_dense/bias:0',
    'generator/mapping/layer_7/equalized_dense/weight:0',
    'generator/mapping/layer_7/equalized_dense/bias:0',
]


def main():
    # prepare generator variables
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]

    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z')
    alpha = tf.Variable(initial_value=0.0, trainable=False, name='transition_alpha')
    fake_images = generator(z, w_dim, n_mapping, alpha, resolutions, featuremaps)

    model_dir = './official-pretrained'
    ckpt_name = 'model.ckpt'
    model_ckpt = os.path.join(model_dir, ckpt_name)

    t_vars = tf.trainable_variables()
    var_list = dict()
    for official, mine in zip(official_code_g_synthesis_t_vars, my_g_synthesis):
        skimmed_official = official.split(':')[0]
        for v in t_vars:
            if v.name == mine:
                var_list[skimmed_official] = v
                break

    for official, mine in zip(official_code_g_mapping_t_vars, my_g_mapping):
        skimmed_official = official.split(':')[0]
        for v in t_vars:
            if v.name == mine:
                var_list[skimmed_official] = v
                break

    saver = tf.train.Saver(var_list=var_list)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        saver.restore(sess, model_ckpt)

        output = sess.run(fake_images, feed_dict={z: np.random.normal(size=[1, z_dim])})
        print(output)
    return


if __name__ == '__main__':
    main()
