import os
import time
import numpy as np
import tensorflow as tf
from utils.utils import compute_shuffle_buffer_size


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


# repeat() with count=None will make dataset repeated indefinitely (managed by tf.estimator's max_step)
# When using dataset.prefetch, use buffer_size=None to let it detect optimal buffer size
def train_input_fn(tfrecord_base_dir, resolution, batch_size, shuffle_buffer_size, epochs=None):
    fn_index = int(np.log2(resolution))
    tfrecord_fn = os.path.join(tfrecord_base_dir, 'ffhq-r{:02d}.tfrecords'.format(fn_index))

    dataset = tf.data.TFRecordDataset(tfrecord_fn)
    dataset = dataset.map(parse_tfrecord_tf, num_parallel_calls=8)
    dataset = dataset.shuffle(buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
    dataset = dataset.repeat(epochs)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=None)
    dataset = dataset.map(
        map_func=lambda images: ({'real_images': images}, tf.constant(0.0)),
        num_parallel_calls=8
    )

    return dataset


# dumy input function --> not used
def eval_input_fn():
    dataset = tf.data.Dataset.range(1)
    dataset = dataset.batch(1)
    dataset = dataset.map(
        map_func=lambda ii: ({'real_images': ii}, tf.constant(0.0)),
        num_parallel_calls=8
    )

    return dataset


def test_input_fn(tfrecord_dir):
    n_samples = 70000
    is_training = True
    my_ram_size_in_gigabytes = 16
    resolution = 256    # 8

    if is_training:
        batch_size = 8
        # compute shuffle buffer size based on ram size
        shuffle_buffer_size = compute_shuffle_buffer_size(my_ram_size_in_gigabytes, resolution, n_samples)
        print('{}x{}: shuffle_buffer_size: {}'.format(resolution, resolution, shuffle_buffer_size))

        dataset = train_input_fn(tfrecord_dir, resolution, batch_size, shuffle_buffer_size)
    else:
        dataset = eval_input_fn()

    iterator = dataset.make_one_shot_iterator()
    images, labels = iterator.get_next()

    step = 0
    start = time.time()
    with tf.Session() as sess:
        while True:
            try:
                feature, label = sess.run([images, labels])
                imgs = feature['real_images']
                step += 1
                if step % 1000 == 0:
                    end = time.time()
                    print('[{:05d}/{:07.3f}s]: {}'.format(step, end - start, imgs.shape))
                    start = end
            except tf.errors.OutOfRangeError:
                print('[{:05d}/{:07.3f}s]: {}'.format(step, end - start, imgs.shape))
                print('End of dataset')
                break
    return


def main():
    tfrecord_dir = './tfrecords'
    test_input_fn(tfrecord_dir)
    return


if __name__ == '__main__':
    main()
