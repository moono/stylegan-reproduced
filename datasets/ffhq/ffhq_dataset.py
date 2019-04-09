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


def add_random_latent_z(images, z_dim):
    z = tf.random_normal(shape=[z_dim], dtype=tf.float32)
    return images, z


def train_input_fn(tfrecord_base_dir, z_dim, resolution, batch_size):
    n_samples = 70000
    fn_index = int(np.log2(resolution))

    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))

    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls=8)
    dataset = dataset.map(lambda images: add_random_latent_z(images, z_dim), num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=n_samples).repeat()
    dataset = dataset.prefetch(batch_size)
    dataset = dataset.batch(batch_size)

    # make dataset as dicionary for features
    dataset = dataset.map(
        map_func=lambda images, z: (
            {
                'real_images': images,
                'z': z
            },
            tf.constant(0.0)),
        num_parallel_calls=8
    )

    return dataset


def eval_input_fn(z_dim, resolution):
    n_samples = 5
    rnd = np.random.RandomState(5)
    z_data = rnd.randn(n_samples, z_dim)

    dataset = tf.data.Dataset.from_tensor_slices(z_data)
    dataset = dataset.map(lambda z: tf.cast(z, dtype=tf.float32), num_parallel_calls=8)
    dataset = dataset.batch(1)
    dataset = dataset.map(
        map_func=lambda z: (
            {
                'real_images': tf.constant(127.5, dtype=tf.float32, shape=[3, resolution, resolution]),
                'z': z,
            }, tf.constant(0, dtype=tf.float32, shape=[])),
        num_parallel_calls=8
    )

    return dataset


def main():
    is_training = False
    z_dim = 512
    resolution = 4
    if is_training:
        tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
        batch_size = 4
        dataset = train_input_fn(tfrecord_dir, z_dim, resolution, batch_size)
    else:
        dataset = eval_input_fn(z_dim, resolution)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                feature, label = sess.run([images, labels])
                imgs = feature['images']
                z = feature['z']
                print()
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


if __name__ == '__main__':
    main()
