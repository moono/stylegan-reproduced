import os
import tensorflow as tf


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)
    })
    data = tf.decode_raw(features['data'], tf.uint8)
    return tf.reshape(data, features['shape'])


def input_fn(tfrecord_base_dir, power, epochs, batch_size, is_training):
    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(power))

    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    return


def main():
    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    input_fn(tfrecord_dir, power=2, epochs=1, batch_size=1, is_training=True)
    return


if __name__ == '__main__':
    main()
