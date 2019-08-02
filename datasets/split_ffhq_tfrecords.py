import os
import glob
import shutil
import tensorflow as tf


# ref: https://stackoverflow.com/questions/54519309/split-tfrecords-file-into-many-tfrecords-files
def split_tfrecord(original_tfrecord_fn, output_dir, n_records, split_size):
    # prepare
    file_size_in_gb = os.path.getsize(original_tfrecord_fn) / (1024.0 * 1024.0 * 1024.0)

    # just copy existing file to new location
    if file_size_in_gb < 1.0:
        shutil.copy(original_tfrecord_fn, output_dir)
        return

    # compute batch size
    batch_size = n_records // split_size
    if n_records % split_size != 0:
        split_size += 1

    format_digits = '0{:d}d'.format(len(str(split_size)))
    with tf.Graph().as_default(), tf.Session() as sess:
        dataset = tf.data.TFRecordDataset(original_tfrecord_fn).batch(batch_size)
        batch = dataset.make_one_shot_iterator().get_next()
        part_num = 0
        while True:
            try:
                print('[{}/{}] creating...'.format(part_num, split_size))
                records = sess.run(batch)
                out_tfrecord_fn = os.path.join(output_dir, '{:s}.tfrecords'.format(format(part_num, format_digits)))
                with tf.python_io.TFRecordWriter(out_tfrecord_fn) as writer:
                    for record in records:
                        writer.write(record)
                part_num += 1
            except tf.errors.OutOfRangeError:
                break

    return


def main():
    tfrecord_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq'
    tfrecord_fns = glob.glob(os.path.join(tfrecord_dir, 'ffhq-*.tfrecords'))
    tfrecord_fns = sorted(tfrecord_fns)

    n_records = 70000
    split_size = 100
    output_base_dir = '/mnt/vision-nas/data-sets/stylegan/ffhq-dataset/tfrecords/ffhq-sliced'

    for cur_res_fn in tfrecord_fns:
        out_dir_name = os.path.basename(os.path.normpath(cur_res_fn))
        output_dir = os.path.join(output_base_dir, out_dir_name)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print('Splitting: {}'.format(out_dir_name))
        split_tfrecord(cur_res_fn, output_dir, n_records, split_size)
    return


if __name__ == '__main__':
    main()
