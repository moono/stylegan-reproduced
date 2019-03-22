import os
import tensorflow as tf
from tensorflow.python.tools import inspect_checkpoint as chkp


def main():
    chpt_fn = os.path.join('./official-pretrained', 'model.ckpt')

    # print all tensors in checkpoint file
    chkp.print_tensors_in_checkpoint_file(chpt_fn, tensor_name='', all_tensors=False)
    return


if __name__ == '__main__':
    main()
