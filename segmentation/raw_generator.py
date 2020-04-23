import glob
import os.path
import numpy as np

from PIL import Image

import tensorflow as tf

# FLAGS = tf.compat.v1.flags.FLAGS

# tf.compat.v1.flags.DEFINE_string('original_gt_folder',
#                                  './VOCdevkit/VOC2012/SegmentationClass',
#                                  'Original ground truth annotations.')
#
# tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')
#
# tf.compat.v1.flags.DEFINE_string('output_dir',
#                                  './VOCdevkit/VOC2012/SegmentationClassRaw',
#                                  'folder to save modified ground truth annotations.')
#
original_gt_folder = './dataset/voc2012/SegmentationClass'
output_dir = './dataset/voc2012/SegmentationClassRaw'


def _remove_colormap(filename):
    """Removes the color map from the annotation.
    Args:
    filename: Ground truth annotation filename.
    Returns:
    Annotation without color map.
    """
    return np.array(Image.open(filename))


def _save_annotation(annotation, filename):
    """Saves the annotation as png file.
    Args:
    annotation: Segmentation annotation.
    filename: Output filename.
    """
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.io.gfile.GFile(filename, mode='w') as f:
        pil_image.save(f, 'PNG')


def main(_):
    # Create the output directory if not exists.
    if not tf.io.gfile.isdir(output_dir):
        tf.io.gfile.makedirs(output_dir)

    annotations = glob.glob(os.path.join(original_gt_folder, '*.' + 'png'))
    for annotation in annotations:
        raw_annotation = _remove_colormap(annotation)
        filename = os.path.basename(annotation)[:-4]
        print(f'{filename} has been saved to {output_dir}')
        _save_annotation(raw_annotation,
                         os.path.join(output_dir, filename + '.' + 'png'))


if __name__ == '__main__':
    tf.compat.v1.app.run()