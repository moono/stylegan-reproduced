import os
import glob
import tensorflow as tf


# 70,000 images
def inspect_ffhq_datasets():
    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    tfrecord_fns = glob.glob(os.path.join(tfrecord_dir, 'ffhq-*.tfrecords'))
    tfrecord_fns = sorted(tfrecord_fns)

    # for each tfrecords, get first data and print the shape
    for tfrecord in tfrecord_fns:
        # print(os.path.basename(tfrecord))
        n_examples = 0
        for record in tf.python_io.tf_record_iterator(tfrecord):
            # example = tf.train.Example.FromString(record)
            # shape = example.features.feature['shape'].int64_list.value
            # data = example.features.feature['data'].bytes_list.value[0]
            # print(shape)
            # print(data)

            n_examples += 1
            # break
        print('{}: {}'.format(os.path.basename(tfrecord), n_examples))
    return


def main():
    inspect_ffhq_datasets()
    return


if __name__ == '__main__':
    main()

