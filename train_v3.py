import os
import argparse
import numpy as np
import tensorflow as tf

from datasets.ffhq.ffhq_dataset import train_input_fn, eval_input_fn
from network_v2.model_fn_v3 import model_fn


tf.logging.set_verbosity(tf.logging.INFO)


# ======================================================================================================================
# tf.contrib.distribute.MirroredStrategy():
# If you are batching your input data, we will place one batch on each GPU in each step.
# So your effective batch size will be num_gpus * batch_size.
# Therefore, consider adjusting your learning rate or batch size according to the number of GPUs

# somehow using distribute strategy with tf.cond() to change / reset optimizer state causes error.
# with single gpu it works fine but with multiple gpus it throws erro
# ======================================================================================================================

# global program arguments parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_base_dir', default='/mnt/vision-nas/moono/trained_models/stylegan-reproduced', type=str)
parser.add_argument('--tfrecord_dir', default='/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq', type=str)
parser.add_argument('--my_ram_size_in_gigabytes', default=16, type=int)
parser.add_argument('--resume_res', default=None, type=int)
args = vars(parser.parse_args())


def get_vars_to_restore(res_to_restore, add_global_step=False):
    vars_list = list()
    if add_global_step:
        vars_list.append('global_step')
    vars_list.append('w_avg')
    vars_list.append('g_mapping/*')

    for r in res_to_restore:
        vars_list.append('g_synthesis/{:d}x{:d}/*'.format(r, r))
    for r in res_to_restore:
        vars_list.append('discriminator/{:d}x{:d}/*'.format(r, r))
    return vars_list


def set_training_ws(res_to_restore, model_base_dir, add_global_step=False):
    res = res_to_restore[-1]
    ws_dir = os.path.join(model_base_dir, '{:d}x{:d}'.format(res, res))
    vars_to_warm_start = get_vars_to_restore(res_to_restore, add_global_step)
    ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir, vars_to_warm_start=vars_to_warm_start)
    return ws


def train(model_dir, tfrecord_dir, train_res, n_images, estimator_params, estimator_ws):
    my_ram_size_in_gigabytes = args['my_ram_size_in_gigabytes']
    z_dim = estimator_params['z_dim']
    batch_size = estimator_params['batch_size']

    # create estimator with distribution training ready
    distribution = tf.contrib.distribute.MirroredStrategy()
    run_config = tf.estimator.RunConfig(keep_checkpoint_max=1,
                                        save_checkpoints_steps=2000,
                                        train_distribute=distribution)
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=model_dir,
        config=run_config,
        params=estimator_params,
        warm_start_from=estimator_ws
    )

    # compute max training step for this resolution
    max_steps = int(np.ceil(n_images / batch_size))

    # start training...
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(tfrecord_dir, z_dim, train_res, batch_size, my_ram_size_in_gigabytes),
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


def main():
    # global args
    model_base_dir = args['model_base_dir']
    tfrecord_dir = args['tfrecord_dir']
    resume_res = args['resume_res']
    model_base_dir = './models'
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
    train_fixed_images_per_res = 60000
    train_trans_images_per_res = 60000
    train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}
    batch_size_base = 2
    learning_rate_base = 0.001
    batch_sizes = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4, 1024: 4}
    g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}

    # find starting resolution for training
    start_train_index = resolutions.index(start_res)
    for ii, train_res in enumerate(resolutions[start_train_index:]):
        if resume_res is not None and train_res < resume_res:
            continue

        do_train_trans = train_with_trans.get(train_res, True)
        print('train_res: {}x{} with transition {}'.format(train_res, train_res, do_train_trans))

        # new resolutions & featuremaps
        original_train_res_index = resolutions.index(train_res)
        train_resolutions = resolutions[:original_train_res_index + 1]
        train_featuremaps = featuremaps[:original_train_res_index + 1]

        # get current batch size
        batch_size = batch_sizes.get(train_res, batch_size_base)

        # set model checkpoint saving locations
        model_dir = os.path.join(model_base_dir, '{:d}x{:d}'.format(train_res, train_res))

        # estimator params
        estimator_params = {
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
            'do_train_trans': do_train_trans,
            'train_trans_images_per_res': train_trans_images_per_res,
            'batch_size': batch_size,
            'g_learning_rate': g_learning_rates.get(train_res, learning_rate_base),
            'd_learning_rate': d_learning_rates.get(train_res, learning_rate_base),
        }

        # determine which variables to warmstart from
        prv_res_to_restore = train_resolutions[:-1]
        cur_res_to_restore = train_resolutions

        # transition training
        print('transition training')
        n_images = train_trans_images_per_res
        ws = None if ii == 0 else set_training_ws(prv_res_to_restore, model_base_dir, add_global_step=False)
        train(model_dir, tfrecord_dir, train_res, n_images, estimator_params, ws)

        # fixed training
        print('fixed training')
        n_images += train_fixed_images_per_res
        ws = set_training_ws(cur_res_to_restore, model_base_dir, add_global_step=True)
        train(model_dir, tfrecord_dir, train_res, n_images, estimator_params, ws)
    return


if __name__ == '__main__':
    main()
