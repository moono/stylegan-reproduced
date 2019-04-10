import os
import numpy as np
import tensorflow as tf

from datasets.ffhq.ffhq_dataset import train_input_fn, eval_input_fn
from network_v2.model_fn import model_fn


tf.logging.set_verbosity(tf.logging.INFO)


def get_vars_to_restore(res_to_restore):
    vars_list = list()
    vars_list.append('w_avg')
    vars_list.append('g_mapping/*')

    for r in res_to_restore:
        vars_list.append('g_synthesis/{:d}x{:d}/*'.format(r, r))
    for r in res_to_restore:
        vars_list.append('discriminator/{:d}x{:d}/*'.format(r, r))
    return vars_list


def main():
    model_save_base_dir = '/mnt/vision-nas/moono/trained_models/stylegan-reproduced'
    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    # model_save_base_dir = './models'
    # tfrecord_dir = './datasets/ffhq/tfrecords'

    # network specific parameters
    z_dim = 512
    w_dim = 512
    n_mapping = 8
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]
    w_ema_decay = 0.995
    style_mixing_prob = 0.9
    truncation_psi = 0.7
    truncation_cutoff = 8

    # training specific parameters
    start_res = 8
    # final_res = resolutions[-1]
    # total_images = 70000
    train_fixed_images_per_res = 600000
    train_trans_images_per_res = 600000
    batch_size_base = 2
    learning_rate_base = 0.001
    batch_sizes = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4, 512: 2, 1024: 2}
    g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

    # find starting resolution for training
    start_train_index = resolutions.index(start_res)
    for ii, train_res in enumerate(resolutions[start_train_index:]):
        print('train_res: {}x{}'.format(train_res, train_res))

        # new resolutions & featuremaps
        original_train_res_index = resolutions.index(train_res)
        train_resolutions = resolutions[:original_train_res_index + 1]
        train_featuremaps = featuremaps[:original_train_res_index + 1]

        # get current batch size
        batch_size = batch_sizes.get(train_res, batch_size_base)

        # set model checkpoint saving locations
        model_dir = os.path.join(model_save_base_dir, '{:d}x{:d}'.format(train_res, train_res))

        # compute max training step for this resolution
        max_steps = int(np.ceil((train_fixed_images_per_res + train_trans_images_per_res) / batch_size))

        # find variables to warm-start for this resolution
        if ii == 0:
            ws = None
        else:
            res_to_restore = resolutions[:resolutions.index(train_res)]
            prev_res = res_to_restore[-1]
            ws_dir = os.path.join(model_save_base_dir, '{:d}x{:d}'.format(prev_res, prev_res))
            vars_to_warm_start = get_vars_to_restore(res_to_restore)
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir, vars_to_warm_start=vars_to_warm_start)

        # create estimator
        distribution = tf.contrib.distribute.MirroredStrategy()
        run_config = tf.estimator.RunConfig(keep_checkpoint_max=1,
                                            save_checkpoints_steps=2000,
                                            train_distribute=distribution)
        model = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=run_config,
            params={
                # generator params
                'z_dim': z_dim,
                'w_dim': w_dim,
                'n_mapping': n_mapping,
                'w_ema_decay': w_ema_decay,
                'style_mixing_prob': style_mixing_prob,
                'truncation_psi': truncation_psi,
                'truncation_cutoff': truncation_cutoff,

                # additional training params
                'resolutions': train_resolutions,
                'featuremaps': train_featuremaps,
                'train_trans_images_per_res': train_trans_images_per_res,
                'batch_size': batch_size,
                'g_learning_rate': g_learning_rates.get(train_res, learning_rate_base),
                'd_learning_rate': d_learning_rates.get(train_res, learning_rate_base),
            },
            warm_start_from=ws
        )

        # start training...
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: train_input_fn(tfrecord_dir, z_dim, train_res, batch_size),
            max_steps=max_steps,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: eval_input_fn(z_dim, train_res),
            steps=10000,
            start_delay_secs=60 * 2,
            throttle_secs=60 * 5,
        )

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    main()
