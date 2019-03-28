import os
import numpy as np
import tensorflow as tf


def parse_tfrecord_tf(record):
    features = tf.parse_single_example(record, features={
        'shape': tf.FixedLenFeature([3], tf.int64),
        'data': tf.FixedLenFeature([], tf.string)
    })

    # [0 ~ 255] uint8
    images = tf.decode_raw(features['data'], tf.uint8)
    images = tf.reshape(images, features['shape'])

    # [0.0 ~ 255.0] float32
    images = tf.cast(images, tf.float32)
    return images


def input_fn(tfrecord_base_dir, resolution, batch_size, is_training):
    n_samples = 70000
    fn_index = int(np.log2(resolution))

    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))

    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls=8)
    # dataset = dataset.apply(tf.contrib.data.map_and_batch(
    #     map_func=lambda record: parse_tfrecord_tf(record),
    #     batch_size=batch_size,
    #     num_parallel_batches=8,
    #     num_parallel_calls=None
    # ))
    if is_training:
        dataset = dataset.shuffle(buffer_size=n_samples).repeat()

    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)

    # make dataset as dicionary for features
    dataset = dataset.map(
        map_func=lambda records: ({'images': records}, tf.constant(0.0)),
        num_parallel_calls=8
    )

    return dataset


def main():
    # tf.enable_eager_execution()

    # for images in dataset:
    #     print()

    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    resolution = 4
    epochs = 1
    batch_size = 1
    is_training = True

    dataset = input_fn(tfrecord_dir, resolution, batch_size, is_training)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                feature, label = sess.run([images, labels])
                image = feature['images']
                print()
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


if __name__ == '__main__':
    main()
