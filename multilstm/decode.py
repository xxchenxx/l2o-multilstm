import tensorflow as tf

dataset = tf.data.TFRecordDataset(['e2e_train.tfrecord'])


def _parse_image_function(example_proto):
  feature_description = {
    'input': tf.io.FixedLenFeature([512], tf.int64),
    'target': tf.io.FixedLenFeature([512], tf.int64),
    'mask': tf.io.FixedLenFeature([512], tf.float32),
  }

  # Parse the input tf.Example proto using the dictionary above.
  example = tf.io.parse_single_example(example_proto, feature_description)
  input_ = tf.reshape(example['input'], [1, -1])
  target = tf.reshape(example['target'], [1, -1])
  mask = tf.reshape(example['mask'], [1, -1])
  return input_, target, mask

dataset = dataset.map(_parse_image_function)
for a,b,c in dataset:
  print(a)