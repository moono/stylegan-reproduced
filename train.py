import os
import numpy as np
import tensorflow as tf

from datasets.ffhq.ffhq_dataset import input_fn
from network.model_fn import model_fn


def get_vars_to_restore(res):
    vars_list = list()
    vars_list.append('w_avg')
    vars_list.append('g_mapping/*')

    res_to_restore = [2 ** r for r in range(int(np.log2(res)), 1, -1)]
    res_to_restore = res_to_restore[::-1]

    for r in res_to_restore:
        vars_list.append('g_synthesis/{:d}x{:d}/*'.format(r, r))
    for r in res_to_restore:
        vars_list.append('discriminator/{:d}x{:d}/*'.format(r, r))
    return vars_list


def main():
    # model_save_base_dir = '/mnt/vision-nas/moono/trained_models/stylegan-reproduced'
    model_save_base_dir = './models'
    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    featuremaps = [512, 512, 512, 512, 256, 128, 64, 32, 16]

    final_res = 1024
    train_start_res = 8
    total_images_in_each_dataset = 70000
    train_fixed_images_per_res = 600000
    train_trans_images_per_res = 600000
    batch_size_base = 2
    learning_rate_base = 0.001
    batch_sizes = {4: 64, 8: 64, 16: 64, 32: 32, 64: 16, 128: 8, 256: 4, 512: 2, 1024: 2}
    g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

    # create estimators
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1, save_checkpoints_steps=10000)
    for ii, res in enumerate(resolutions):
        # get current batch size
        batch_size = batch_sizes.get(res, batch_size_base)

        # set model checkpoint saving locations
        model_dir = os.path.join(model_save_base_dir, '{:d}x{:d}'.format(res, res))

        # find variables to warm-start for this resolution
        if ii == 0:
            ws = None
        else:
            prev_res = resolutions[ii-1]
            ws_dir = os.path.join(model_save_base_dir, '{:d}x{:d}'.format(prev_res, prev_res))
            vars_to_warm_start = get_vars_to_restore(prev_res)
            ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir, vars_to_warm_start=vars_to_warm_start)

        # create estimator
        model = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=model_dir,
            config=run_config,
            params={
                'res': res,
                'final_res': final_res,
                'resolutions': resolutions,
                'featuremaps': featuremaps,
                'total_images_in_each_dataset': total_images_in_each_dataset,
                'train_fixed_images_per_res': train_fixed_images_per_res,
                'train_trans_images_per_res': train_trans_images_per_res,
                'batch_size': batch_size,
                'g_learning_rate': g_learning_rates.get(res, learning_rate_base),
                'd_learning_rate': d_learning_rates.get(res, learning_rate_base)
            },
            warm_start_from=ws
        )

        # start training...
        train_spec = tf.estimator.TrainSpec(
            input_fn=lambda: input_fn(tfrecord_dir, res, batch_size, True),
            max_steps=None,
        )
        eval_spec = tf.estimator.EvalSpec(
            input_fn=lambda: input_fn(tfrecord_dir, res, batch_size, False),
            steps=10000,
            start_delay_secs=60 * 2,
            throttle_secs=60 * 5,
        )

        tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


if __name__ == '__main__':
    main()
