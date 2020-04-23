# codind=utf-8

"""Converts dataset to TFRecord file format with Example protos."""

import math
import os
import random
import sys
import build_data
import tensorflow as tf
import shutil


_NUM_SHARDS = 1  # The number of tfrecord file per dataset split
OUTPUT_DIR = './dataset/tfrecord'  # Path to directory saving tfrecord files


def _convert_dataset(dataset_split, dataset_image_dir, dataset_label_dir):
    """Converts the ADE20k dataset into into tfrecord format.
    Args:
      dataset_split: Dataset split (e.g., train, val).
      dataset_image_dir: Dir in which the images locates.
      dataset_label_dir: Dir in which the annotations locates.
    Raises:
      RuntimeError: If loaded image and label have different shape.
    """

    img_names = tf.io.gfile.glob(os.path.join(dataset_image_dir, '*.jpg'))
    random.shuffle(img_names)
    seg_names = []
    for f in img_names:
        # get the filename without the extension
        basename = os.path.basename(f).split('.')[0]
        # cover its corresponding *_seg.png
        seg = os.path.join(dataset_label_dir, basename + '.png')
        seg_names.append(seg)

    num_images = len(img_names)
    num_per_shard = int(math.ceil(num_images / float(_NUM_SHARDS)))

    for shard_id in range(_NUM_SHARDS):
        output_filename = os.path.join(OUTPUT_DIR,
                                       '%s-%05d-of-%05d.tfrecord' % (dataset_split, shard_id, _NUM_SHARDS))
        with tf.io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_idx = shard_id * num_per_shard
            end_idx = min((shard_id + 1) * num_per_shard, num_images)
            for i in range(start_idx, end_idx):
                sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                    i + 1, num_images, shard_id))
                sys.stdout.flush()

                # read images
                image_filename = img_names[i]
                image_data = open(image_filename, 'rb').read()

                height, width = build_data.ImageReader(image_data, 'jpg', channels=3).read_image_dims()
                # read labels
                seg_filename = seg_names[i]
                seg_data = open(seg_filename, 'rb').read()
                seg_height, seg_width = build_data.ImageReader(seg_data, 'png', channels=3).read_image_dims()
                if height != seg_height or width != seg_width:
                    raise RuntimeError('Shape mismatched between image and label.')
                # Convert to tf example.
                example = build_data.image_seg_to_tfexample(
                    image_data, img_names[i], height, width, seg_data)
                tfrecord_writer.write(example.SerializeToString())
        sys.stdout.write('\n')
        sys.stdout.flush()


def listdir_nohidden(path):
    for f in os.listdir(path):
        if f[-4:] == '.png' or f[-4:] == '.jpg':
            yield f[:-4]


def mkdir(path):
    if not os.path.exists(path):
        mkdir(os.path.dirname(path))
        os.mkdir(path)
    else:
        return


def main():
    train_file = './dataset/voc2012/ImageSets/Segmentation/train.txt'
    val_file = './dataset/voc2012/ImageSets/Segmentation/val.txt'
    train_list = []
    with open(train_file, encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').replace('\r', '')
            train_list.append(line)
    f.close()

    val_list = []
    with open(val_file, encoding='utf-8') as f:
        for line in f:
            line = line.replace('\n', '').replace('\r', '')
            val_list.append(line)
    f.close()

    train_img_path = './dataset/images/train/img'  # Path to the directory containing train set image.
    train_label_path = './dataset/images/train/label'  # Path to the directory containing train set label.
    val_img_path = './dataset/images/val/img'  # Path to the directory containing val set image.
    val_label_path = './dataset/images/val/label'  # Path to the directory containing val set label.

    mkdir(train_img_path)
    mkdir(train_label_path)
    mkdir(val_img_path)
    mkdir(val_label_path)

    img_path = './dataset/voc2012/JPEGImages'
    label_path = './dataset/voc2012/SegmentationClass'

    for file in list(listdir_nohidden(img_path)):
        if file in train_list:
            shutil.copy(os.path.join(img_path, file + '.jpg'), os.path.join(train_img_path, file + '.jpg'))
            shutil.copy(os.path.join(label_path, file + '.png'), os.path.join(train_label_path, file + '.png'))
            print(f'copy {file} to trainset!')
        elif file in val_list:
            shutil.copy(os.path.join(img_path, file + '.jpg'), os.path.join(val_img_path, file + '.jpg'))
            shutil.copy(os.path.join(label_path, file + '.png'), os.path.join(val_label_path, file + '.png'))
            print(f'copy {file} to valset!')
        else:
            continue

    _convert_dataset('train', train_img_path, train_label_path)
    _convert_dataset('val', val_img_path, val_label_path)

if __name__ == '__main__':
    tf.compat.v1.app.run()
th, file + '.png'))
            print(f'copy {file} to valset!')
        else:
            continue

    _convert_dataset('train', train_img_path, train_label_path)
    _convert_dataset('val', val_img_path, val_label_path)

if __name__ == '__main__':
    tf.compat.v1.app.run()
