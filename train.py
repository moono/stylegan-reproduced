import os
import argparse
import numpy as np
import tensorflow as tf

from datasets.ffhq.ffhq_dataset import train_input_fn, eval_input_fn
from network.model_fn import model_fn
from utils.utils import compute_shuffle_buffer_size


tf.logging.set_verbosity(tf.logging.INFO)

# ======================================================================================================================
# tf.contrib.distribute.MirroredStrategy():
# If you are batching your input data, we will place one batch on each GPU in each step.
# So your effective batch size will be num_gpus * batch_size.
# Therefore, consider adjusting your learning rate or batch size according to the number of GPUs

# somehow using distribute strategy with tf.cond() to change / reset optimizer state causes error.
# with single gpu it works fine but with multiple gpus it throws error
# ======================================================================================================================

# global program arguments parser
parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_base_dir', default='./models', type=str)
parser.add_argument('--tfrecord_dir', default='/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq', type=str)
parser.add_argument('--my_ram_size_in_gigabytes', default=16, type=int)
parser.add_argument('--n_samples', default=70000, type=int)
args = vars(parser.parse_args())


# exclude optimizer variables
def get_vars_to_restore(res_to_restore, add_global_step=False):
    vars_list = list()
    if add_global_step:
        vars_list.append('global_step')
    vars_list.append('w_avg')
    vars_list.append('^(?=.*(?:g_mapping))(?!.*(?:Adam)).*$')

    for r in res_to_restore:
        regex_str = '^(?=.*g_synthesis\/{:d}x{:d})(?!.*(?:Adam)).*$'.format(r, r)
        vars_list.append(regex_str)
    for r in res_to_restore:
        regex_str = '^(?=.*discriminator\/{:d}x{:d})(?!.*(?:Adam)).*$'.format(r, r)
        vars_list.append(regex_str)
    return vars_list


def set_training_ws(res_to_restore, model_base_dir, add_global_step=False):
    res = res_to_restore[-1]
    ws_dir = os.path.join(model_base_dir, '{:d}x{:d}'.format(res, res))

    # check if model directory exists
    if os.path.exists(ws_dir):
        vars_to_warm_start = get_vars_to_restore(res_to_restore, add_global_step)
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=ws_dir, vars_to_warm_start=vars_to_warm_start)
    else:
        ws = None
    return ws


def train(model_dir, train_res, n_images, estimator_params, estimator_ws):
    # early exit condition
    if n_images <= 0:
        return

    # fetch parameters
    tfrecord_dir = args['tfrecord_dir']
    my_ram_size_in_gigabytes = args['my_ram_size_in_gigabytes']
    n_samples = args['n_samples']
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

    # compute shuffle buffer size based on ram size
    shuffle_buffer_size = compute_shuffle_buffer_size(my_ram_size_in_gigabytes, train_res, n_samples)
    tf.logging.log(tf.logging.INFO,
                   '[moono]: {}x{}: shuffle_buffer_size: {}'.format(train_res, train_res, shuffle_buffer_size))

    # start training...
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: train_input_fn(tfrecord_dir, train_res, batch_size, shuffle_buffer_size),
        max_steps=max_steps,
    )
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: eval_input_fn(),
        steps=10000,
        start_delay_secs=60 * 2,
        throttle_secs=60 * 5,
    )

    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)
    return


def compute_training_image_counts(train_start_res, resolutions, train_with_trans,
                                  train_trans_images_per_res, train_fixed_images_per_res, train_total_n_images):
    cur_image_count = 0
    train_n_images = dict()
    for res, do_transition in train_with_trans.items():
        if res < train_start_res:
            n_trans = 0
            n_fixed = 0
        else:
            if res != resolutions[-1]:
                n_trans = train_trans_images_per_res if do_transition else 0
                n_fixed = train_fixed_images_per_res
            else:
                n_trans = train_trans_images_per_res if do_transition else 0
                n_fixed = train_total_n_images - (cur_image_count + n_trans)

        train_n_images[res] = {
            'trans': n_trans,
            'fixed': n_fixed,
            # 'total': n_trans + n_fixed
        }

        # update
        cur_image_count = cur_image_count + train_n_images[res]['trans'] + train_n_images[res]['fixed']

    double_check = [v['trans'] + v['fixed'] for k, v in train_n_images.items()]
    assert sum(double_check) == train_total_n_images
    return train_n_images


def main():
    # global args
    model_base_dir = args['model_base_dir']

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
    train_start_res = 8
    train_total_n_images = 25000000
    train_trans_images_per_res = 600000
    train_fixed_images_per_res = 600000
    batch_size_base = 2
    learning_rate_base = 0.001
    batch_sizes = {4: 128, 8: 128, 16: 128, 32: 64, 64: 32, 128: 16, 256: 8, 512: 4, 1024: 4}
    g_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    d_learning_rates = {128: 0.0015, 256: 0.002, 512: 0.003, 1024: 0.003}
    train_with_trans = {4: False, 8: False, 16: True, 32: True, 64: True, 128: True, 256: True, 512: True, 1024: True}
    train_n_images = compute_training_image_counts(train_start_res, resolutions, train_with_trans,
                                                   train_trans_images_per_res, train_fixed_images_per_res,
                                                   train_total_n_images)
    # start training
    train_start_res_idx = resolutions.index(train_start_res)
    for ii, res in enumerate(resolutions[train_start_res_idx:]):
        do_train_trans = train_with_trans.get(res, True)

        tf.logging.log(tf.logging.INFO, '[moono]: train_res: {}x{} with transition {}'.format(res, res, do_train_trans))

        # new resolutions & featuremaps
        original_train_resolution_index = resolutions.index(res)
        train_resolutions = resolutions[:original_train_resolution_index + 1]
        train_featuremaps = featuremaps[:original_train_resolution_index + 1]

        # get current batch size
        batch_size = batch_sizes.get(res, batch_size_base)

        # set model checkpoint saving locations
        model_dir = os.path.join(model_base_dir, '{:d}x{:d}'.format(res, res))

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
            'train_trans_images_per_res': train_n_images[res]['trans'],
            'batch_size': batch_size,
            'g_learning_rate': g_learning_rates.get(res, learning_rate_base),
            'd_learning_rate': d_learning_rates.get(res, learning_rate_base),
        }

        # determine which variables to warmstart from
        prv_res_to_restore = train_resolutions[:-1]
        cur_res_to_restore = train_resolutions

        # transition training (restore variable from previous resolution without global_step)
        tf.logging.log(tf.logging.INFO, '[moono]: transition training')
        ws = None if ii == 0 else set_training_ws(prv_res_to_restore, model_base_dir, add_global_step=False)
        n_images_to_show = train_n_images[res]['trans']
        train(model_dir, res, n_images_to_show, estimator_params, estimator_ws=ws)

        # fixed training (restore variable from current resolution with global_step)
        tf.logging.log(tf.logging.INFO, '[moono]: fixed training')
        ws = set_training_ws(cur_res_to_restore, model_base_dir, add_global_step=True)
        n_images_to_show = train_n_images[res]['trans'] + train_n_images[res]['fixed']
        train(model_dir, res, n_images_to_show, estimator_params, estimator_ws=ws)
    return


if __name__ == '__main__':
    main()
