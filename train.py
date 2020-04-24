from glob import glob
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from deeplab import DeepLabV3Plus
import data_generator


batch_size = 24
H, W = 512, 512
num_classes = 21

train_dataset = data_generator.Dataset(
          dataset_name='pascal_voc_seg',
          split_name='train',
          dataset_dir='./tfrecord/train-00000-of-00001.tfrecord',
          batch_size=batch_size,
          crop_size=[384, 384],
          min_scale_factor=0.5,
          max_scale_factor=2,
          scale_factor_step_size=0.25,
          model_variant='resnet_v1_50',
          num_readers=4,
          is_training=True,
          should_shuffle=True,
          should_repeat=True)

val_dataset = data_generator.Dataset(
          dataset_name='pascal_voc_seg',
          split_name='val',
          dataset_dir='./tfrecord/val-00000-of-00001.tfrecord',
          batch_size=batch_size,
          crop_size=[384, 384],
          min_scale_factor=0.5,
          max_scale_factor=2,
          scale_factor_step_size=0.25,
          model_variant='resnet_v1_50',
          num_readers=4,
          is_training=True,
          should_shuffle=True,
          should_repeat=True)


loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)
strategy = tf.distribute.MirroredStrategy()
with strategy.scope():
    model = DeepLabV3Plus(H, W, num_classes)
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = 0.9997
            layer.epsilon = 1e-5
        elif isinstance(layer, tf.keras.layers.Conv2D):
            layer.kernel_regularizer = tf.keras.regularizers.l2(1e-4)
    model.compile(loss=loss,
                  optimizer=tf.optimizers.Adam(learning_rate=1e-4),
                  metrics=['accuracy'])


tb = TensorBoard(log_dir='logs', write_graph=True, update_freq='batch')
mc = ModelCheckpoint(mode='min', filepath='top_weights.h5',
                     monitor='val_loss',
                     save_best_only='True',
                     save_weights_only='True', verbose=1)
callbacks = [mc, tb]

print(type(train_dataset))
model.fit(train_dataset,
          steps_per_epoch=10582 // batch_size,
          epochs=300,
          validation_data=val_dataset,
          validation_steps=1449 // batch_size,
          callbacks=callbacks)
