import os
import numpy as np
import tensorflow as tf


# n_samples = 70000
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


def compute_shuffle_buffer_size(my_ram_size_in_gigabytes, resolution):
    image_shape = (3, resolution, resolution)
    uint8_in_bytes = np.dtype('uint8').itemsize
    bytes_per_image = np.prod(image_shape) * uint8_in_bytes
    buffer_size_limit_in_gigabytes = max(1, my_ram_size_in_gigabytes // 8)
    buffer_size_limit_in_bytes = buffer_size_limit_in_gigabytes * (2 ** 30)
    shuffle_buffer_size = (buffer_size_limit_in_bytes - 1) // bytes_per_image + 1
    return shuffle_buffer_size


def train_input_fn(tfrecord_base_dir, z_dim, resolution, batch_size, my_ram_size_in_gigabytes, epoch=None):
    # compute shuffle buffer size
    n_samples = 70000
    shuffle_buffer_size = compute_shuffle_buffer_size(my_ram_size_in_gigabytes, resolution)
    shuffle_buffer_size = min(shuffle_buffer_size, n_samples + 1)

    fn_index = int(np.log2(resolution))

    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))

    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls=8)
    dataset = dataset.map(lambda images: add_random_latent_z(images, z_dim), num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size)

    # repeat() with count=None will make dataset repeated indefinitely (managed by tf.estimator's max_step)
    dataset = dataset.repeat(epoch)

    # When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size
    dataset = dataset.prefetch(buffer_size=None)
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


def test_input_fn():
    is_training = True
    z_dim = 512
    resolution = 256
    my_ram_size_in_gigabytes = 16

    # in_mb = image_mb * buffer_size
    if is_training:
        tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
        batch_size = 8
        dataset = train_input_fn(tfrecord_dir, z_dim, resolution, batch_size, my_ram_size_in_gigabytes)
    else:
        dataset = eval_input_fn(z_dim, resolution)

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()
    with tf.Session() as sess:
        while True:
            try:
                feature, label = sess.run([images, labels])
                imgs = feature['real_images']
                z = feature['z']
                print()
            except tf.errors.OutOfRangeError:
                print('End of dataset')
                break
    return


def test_memory_overflow():
    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    z_dim = 512

    resolutions = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
    batch_sizes = [128, 128, 128, 64, 32, 16, 8, 4, 4]
    my_ram_size_in_gigabytes = 16

    for res, batch_size in zip(resolutions[:-2], batch_sizes[:-2]):
        dataset = train_input_fn(tfrecord_dir, z_dim, res, batch_size, my_ram_size_in_gigabytes, epoch=1)
        iterator = dataset.make_one_shot_iterator()
        images, labels = iterator.get_next()

        index = 0
        with tf.Session() as sess:
            while True:
                try:
                    feature, label = sess.run([images, labels])
                    imgs = feature['real_images']
                    z = feature['z']

                    if index % 100 == 0:
                        print('image shape: {}'.format(imgs.shape))

                    index += 1
                except tf.errors.OutOfRangeError:
                    print('End of dataset')
                    break
    return


def main():
    test_memory_overflow()
    test_input_fn()
    return


if __name__ == '__main__':
    main()
