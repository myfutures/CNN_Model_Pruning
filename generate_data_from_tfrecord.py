import tensorflow as tf
import scipy.misc
import os
import config as conf

slim = tf.contrib.slim

#读取数据时的系统设置
num_readers=conf.num_readers

#训练数据
resize_side_min=conf.resize_side_min
resize_side_max=conf.resize_side_max
BATCH_SIZE=conf.batch_size
default_image_size=conf.default_image_size                   #原文中是224，但caffe的实现似乎用了227，为了权值读取方便，这里改成了227

#imagenet属性
num_class=conf.imagenet['num_class']
#mean of imagenet dataset in BGR
imagenet_mean = conf.imagenet['mean']                        #google [123.68,116.78,103.94]
# imagenet_mean = np.array([123.68,116.78,103.94], dtype=np.float32)      #google [123.68,116.78,103.94]
label_offset=conf.imagenet['label_offset']



#文件名相关信息
_FILE_PATTERN = '%s-*'
_SPLITS_TO_SIZES = {
    'train': 1281167,
    'validation': 50000,
}
label_filename="/home/victorfang/Desktop/imagenet_tfrecord/labels.txt"
dataset_directory='/home/victorfang/Desktop/imagenet_tfrecord'

_ITEMS_TO_DESCRIPTIONS = {
    'image': 'A color image of varying height and width.',
    'label': 'The label id of the image, integer between 0 and 999',
    'label_text': 'The text of the label.',
    'object/bbox': 'A list of bounding boxes.',
    'object/label': 'A list of labels, one per each object.',
}

def _central_crop(image_list, crop_height, crop_width):
  """Performs central crops of the given image list.

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the height of the image following the crop.
    crop_width: the width of the image following the crop.

  Returns:
    the list of cropped images.
  """
  outputs = []
  for image in image_list:
    image_height = tf.shape(image)[0]
    image_width = tf.shape(image)[1]

    offset_height = (image_height - crop_height) / 2
    offset_width = (image_width - crop_width) / 2

    outputs.append(_crop(image, offset_height, offset_width,
                         crop_height, crop_width))
  return outputs

def _crop(image, offset_height, offset_width, crop_height, crop_width):
  """Crops the given image using the provided offsets and sizes.

  Note that the method doesn't assume we know the input image size but it does
  assume we know the input image rank.

  Args:
    image: an image of shape [height, width, channels].
    offset_height: a scalar tensor indicating the height offset.
    offset_width: a scalar tensor indicating the width offset.
    crop_height: the height of the cropped image.
    crop_width: the width of the cropped image.

  Returns:
    the cropped (and resized) image.

  Raises:
    InvalidArgumentError: if the rank is not 3 or if the image dimensions are
      less than the crop size.
  """
  original_shape = tf.shape(image)

  rank_assertion = tf.Assert(
      tf.equal(tf.rank(image), 3),
      ['Rank of image must be equal to 3.'])
  with tf.control_dependencies([rank_assertion]):
    cropped_shape = tf.stack([crop_height, crop_width, original_shape[2]])

  size_assertion = tf.Assert(
      tf.logical_and(
          tf.greater_equal(original_shape[0], crop_height),
          tf.greater_equal(original_shape[1], crop_width)),
      ['Crop size greater than the image size.'])

  offsets = tf.to_int32(tf.stack([offset_height, offset_width, 0]))

  # Use tf.slice instead of crop_to_bounding box as it accepts tensors to
  # define the crop size.
  with tf.control_dependencies([size_assertion]):
    image = tf.slice(image, offsets, cropped_shape)
  return tf.reshape(image, cropped_shape)

def _random_crop(image_list, crop_height, crop_width):
  """Crops the given list of images.

  The function applies the same crop to each image in the list. This can be
  effectively applied when there are multiple image inputs of the same
  dimension such as:

    image, depths, normals = _random_crop([image, depths, normals], 120, 150)

  Args:
    image_list: a list of image tensors of the same dimension but possibly
      varying channel.
    crop_height: the new height.
    crop_width: the new width.

  Returns:
    the image_list with cropped images.

  Raises:
    ValueError: if there are multiple image inputs provided with different size
      or the images are smaller than the crop dimensions.
  """
  if not image_list:
    raise ValueError('Empty image_list.')

  # Compute the rank assertions.
  rank_assertions = []
  for i in range(len(image_list)):
    image_rank = tf.rank(image_list[i])
    rank_assert = tf.Assert(
        tf.equal(image_rank, 3),
        ['Wrong rank for tensor  %s [expected] [actual]',
         image_list[i].name, 3, image_rank])
    rank_assertions.append(rank_assert)

  with tf.control_dependencies([rank_assertions[0]]):
    image_shape = tf.shape(image_list[0])
  image_height = image_shape[0]
  image_width = image_shape[1]
  crop_size_assert = tf.Assert(
      tf.logical_and(
          tf.greater_equal(image_height, crop_height),
          tf.greater_equal(image_width, crop_width)),
      ['Crop size greater than the image size.'])

  asserts = [rank_assertions[0], crop_size_assert]

  for i in range(1, len(image_list)):
    image = image_list[i]
    asserts.append(rank_assertions[i])
    with tf.control_dependencies([rank_assertions[i]]):
      shape = tf.shape(image)
    height = shape[0]
    width = shape[1]

    height_assert = tf.Assert(
        tf.equal(height, image_height),
        ['Wrong height for tensor %s [expected][actual]',
         image.name, height, image_height])
    width_assert = tf.Assert(
        tf.equal(width, image_width),
        ['Wrong width for tensor %s [expected][actual]',
         image.name, width, image_width])
    asserts.extend([height_assert, width_assert])

  # Create a random bounding box.
  #
  # Use tf.random_uniform and not numpy.random.rand as doing the former would
  # generate random numbers at graph eval time, unlike the latter which
  # generates random numbers at graph definition time.
  with tf.control_dependencies(asserts):
    max_offset_height = tf.reshape(image_height - crop_height + 1, [])
  with tf.control_dependencies(asserts):
    max_offset_width = tf.reshape(image_width - crop_width + 1, [])
  offset_height = tf.random_uniform(
      [], maxval=max_offset_height, dtype=tf.int32)
  offset_width = tf.random_uniform(
      [], maxval=max_offset_width, dtype=tf.int32)

  return [_crop(image, offset_height, offset_width,
                crop_height, crop_width) for image in image_list]

def _smallest_size_at_least(height, width, smallest_side):
  """Computes new shape with the smallest side equal to `smallest_side`.

  Computes new shape with the smallest side equal to `smallest_side` while
  preserving the original aspect ratio.

  Args:
    height: an int32 scalar tensor indicating the current height.
    width: an int32 scalar tensor indicating the current width.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    new_height: an int32 scalar tensor indicating the new height.
    new_width: and int32 scalar tensor indicating the new width.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  height = tf.to_float(height)
  width = tf.to_float(width)
  smallest_side = tf.to_float(smallest_side)

  scale = tf.cond(tf.greater(height, width),
                  lambda: smallest_side / width,
                  lambda: smallest_side / height)
  new_height = tf.to_int32(tf.rint(height * scale))
  new_width = tf.to_int32(tf.rint(width * scale))
  return new_height, new_width

def read_label_file(labels_filename):
  """Reads the labels file and returns a mapping from ID to class name.

  Args:
    dataset_dir: The directory in which the labels file is found.
    filename: The filename where the class names are written.

  Returns:
    A map from a label (integer) to class name.
  """
  with tf.gfile.Open(labels_filename, 'rb') as f:
    lines = f.read().decode()
  lines = lines.split('\n')
  lines = filter(None, lines)

  labels_to_class_names = {}
  for line in lines:
    index = line.index(':')
    labels_to_class_names[int(line[:index])] = line[index+1:]
  return labels_to_class_names

def _aspect_preserving_resize(image, smallest_side):
  """Resize images preserving the original aspect ratio纵横比.

  Args:
    image: A 3-D image `Tensor`.
    smallest_side: A python integer or scalar `Tensor` indicating the size of
      the smallest side after resize.

  Returns:
    resized_image: A 3-D tensor containing the resized image.
  """
  smallest_side = tf.convert_to_tensor(smallest_side, dtype=tf.int32)

  shape = tf.shape(image)
  height = shape[0]
  width = shape[1]
  new_height, new_width = _smallest_size_at_least(height, width, smallest_side)
  image = tf.expand_dims(image, 0)
  resized_image = tf.image.resize_bilinear(image, [new_height, new_width],
                                           align_corners=False)
  resized_image = tf.squeeze(resized_image)
  resized_image.set_shape([None, None, 3])
  return resized_image

def _mean_image_subtraction(image, means):
  """Subtracts the given means from each image channel.

  For example:
    means = [123.68, 116.779, 103.939]
    image = _mean_image_subtraction(image, means)

  Note that the rank of `image` must be known.

  Args:
    image: a tensor of size [height, width, C].
    means: a C-vector of values to subtract from each channel.

  Returns:
    the centered image.

  Raises:
    ValueError: If the rank of `image` is unknown, if `image` has a rank other
      than three or if the number of channels in `image` doesn't match the
      number of values in `means`.
  """
  if image.get_shape().ndims != 3:
    raise ValueError('Input must be of size [height, width, C>0]')
  num_channels = image.get_shape().as_list()[-1]
  if len(means) != num_channels:
    raise ValueError('len(means) must match the number of channels')

  channels = tf.split(axis=2, num_or_size_splits=num_channels, value=image)
  for i in range(num_channels):
    channels[i] -= means[i]
  return tf.concat(axis=2, values=channels)

def read_tfrecord(split_name, dataset_dir, file_pattern=None,batch_size=BATCH_SIZE):
    if split_name not in _SPLITS_TO_SIZES:
        raise ValueError('split name %s was not recognized.' % split_name)

    if not file_pattern:
        file_pattern = _FILE_PATTERN
    file_pattern = os.path.join(dataset_dir, file_pattern % split_name)

    keys_to_features = {
        'image/encoded': tf.FixedLenFeature(
            (), tf.string, default_value=''),
        'image/format': tf.FixedLenFeature(
            (), tf.string, default_value='jpeg'),
        'image/class/label': tf.FixedLenFeature(
            [], dtype=tf.int64, default_value=-1),
        'image/class/text': tf.FixedLenFeature(
            [], dtype=tf.string, default_value=''),
        'image/object/bbox/xmin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymin': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/xmax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/bbox/ymax': tf.VarLenFeature(
            dtype=tf.float32),
        'image/object/class/label': tf.VarLenFeature(
            dtype=tf.int64),
    }

    items_to_handlers = {
        'image': slim.tfexample_decoder.Image('image/encoded', 'image/format'),
        'label': slim.tfexample_decoder.Tensor('image/class/label'),
        'label_text': slim.tfexample_decoder.Tensor('image/class/text'),
        'object/bbox': slim.tfexample_decoder.BoundingBox(
            ['ymin', 'xmin', 'ymax', 'xmax'], 'image/object/bbox/'),
        'object/label': slim.tfexample_decoder.Tensor('image/object/class/label'),
    }

    decoder = slim.tfexample_decoder.TFExampleDecoder(
        keys_to_features, items_to_handlers)
    labels_to_names = read_label_file(label_filename)
    dataset = slim.dataset.Dataset(
        data_sources=file_pattern,
        reader=tf.TFRecordReader,
        decoder=decoder,
        num_samples=_SPLITS_TO_SIZES[split_name],  # 训练数据的总数
        items_to_descriptions=_ITEMS_TO_DESCRIPTIONS,
        num_classes=num_class,
        labels_to_names=labels_to_names  # 字典形式，格式为：id:class_call,
    )

    provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        num_readers=num_readers,
        common_queue_capacity=2 * batch_size,
        common_queue_min= batch_size,
        shuffle=True
    )
    [image, label,label_text] = provider.get(['image', 'label','label_text'])

    #对图像和标签进行处理
    label -= label_offset
    train_image_size = default_image_size
    resize_side = tf.random_uniform(
        [], minval=resize_side_min, maxval=resize_side_max + 1, dtype=tf.int32)
    image = _aspect_preserving_resize(image, resize_side)                       #把图像在保存横款比的情况下缩放到给定范围
    # image = _aspect_preserving_resize(image, 227)  # 把图像在保存横款比的情况下缩放到给定范围
    if split_name is 'train':
        image = _random_crop([image], train_image_size, train_image_size)[0]        #训练时，随机裁剪一部分图像作为输入
    else:
        image = _central_crop([image], train_image_size, train_image_size)[0]      #evaluate时，以图像中心参见一部分作为输入。必要性？
    image.set_shape([train_image_size,train_image_size,3])
    image=tf.to_float(image)


    if split_name is 'train':
        image = tf.image.random_flip_left_right(image)                              #训练时，将图像随机左右翻转

    _mean_image_subtraction(image, imagenet_mean)                 #使用时需要把每张图的rgb减去各自的平均值，详见vgg_processing.py

    images, labels ,labels_text= tf.train.batch(
        [image, label,label_text],
        batch_size=batch_size,
        num_threads=4,
        capacity=5 * batch_size,
    )

    #标签进行01编码
    labels = slim.one_hot_encoding(
        labels, num_class-label_offset)

    # batch_queue = slim.prefetch_queue.prefetch_queue(
    #     [images, labels], capacity=2 * deploy_config.num_clones)

    return images, labels,labels_text

def main():
    img, label ,labels_text= read_tfrecord('train',dataset_directory)

    init = tf.initialize_all_variables()
    local_init=tf.initialize_local_variables()
    coord = tf.train.Coordinator()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(local_init)

        threads = tf.train.start_queue_runners(sess, coord=coord)
        for i in range(9):
            val,l,l_text= sess.run([img, label,labels_text])
            # val, l = sess.run([img_batch, label_batch])
            print(l_text[0])
            scipy.misc.imsave('outfile.jpg', val[0])
        coord.request_stop()
        coord.join(threads)


           # i=val[0].eval
            #我们也可以根据需要对val， l进行处理
            #l = to_categorical(l, 12)
            #print(val.shape, l)

if __name__ == '__main__':
    main()