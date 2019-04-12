import os
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp


def main():
    # ckpt_fn = os.path.join('../official-pretrained', 'model.ckpt')
    ckpt_fn = tf.train.latest_checkpoint('../models/8x8')

    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file(ckpt_fn, tensor_name='', all_tensors=False)
    return


if __name__ == '__main__':
    main()
