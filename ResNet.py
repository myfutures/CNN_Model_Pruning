import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.training import moving_averages

def inference(x, is_training,
              num_classes=1000,
              num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
              use_bias=False,  # defaults to using batch norm
              bottleneck=True):
    c = dict()
    c['bottleneck'] = bottleneck
    c['is_training'] = tf.convert_to_tensor(is_training,
                                            dtype='bool',
                                            name='is_training')
    c['ksize'] = 3
    c['stride'] = 1
    c['use_bias'] = use_bias
    c['fc_units_out'] = num_classes
    c['num_blocks'] = num_blocks
    c['stack_stride'] = 2
    with tf.variable_scope('resnet_v1_50'):
        with tf.variable_scope('conv1'):
            c['conv_filters_out'] = 64
            c['ksize'] = 7
            c['stride'] = 2
            x = conv(x, c)
            x = bn(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('block1'):
            x = max_pool(x, ksize=3, stride=2)
            c['num_blocks'] = num_blocks[0]
            c['stack_stride'] = 1
            c['block_filters_internal'] = 64                                    #第一个卷积核的数量，即输出几通道
            x = stack(x, c)

        with tf.variable_scope('block2'):
            c['num_blocks'] = num_blocks[1]
            c['block_filters_internal'] = 128
            c['stack_stride'] = 2
            x = stack(x, c)

        with tf.variable_scope('block3'):
            c['num_blocks'] = num_blocks[2]
            c['block_filters_internal'] = 256
            x = stack(x, c)

        with tf.variable_scope('block4'):
            c['num_blocks'] = num_blocks[3]
            c['block_filters_internal'] = 512
            x = stack(x, c)

        # post-net
        x = tf.reduce_mean(x, reduction_indices=[1, 2], name="avg_pool")

        if num_classes != None:
            with tf.variable_scope('logits'):
                x = fc(x, c)
    return x

def conv(x, c):
    ksize = c['ksize']
    stride = c['stride']
    filters_out = c['conv_filters_out']

    filters_in = x.get_shape()[-1]
    shape = [ksize, ksize, filters_in, filters_out]
    initializer = tf.truncated_normal_initializer(stddev=0.00004)
    weights = get_variable('weights',
                            shape=shape,
                            dtype='float',
                            initializer=initializer,
                            weight_decay=0.1)
    return tf.nn.conv2d(x, weights, [1, stride, stride, 1], padding='SAME')

def bn(x, c):
    x_shape = x.get_shape()
    params_shape = x_shape[-1:]

    with tf.variable_scope('BatchNorm'):

        if c['use_bias']:
            bias = get_variable('bias', params_shape,
                                 initializer=tf.zeros_initializer)
            return x + bias

        axis = list(range(len(x_shape) - 1))

        beta = get_variable('beta',
                             params_shape,
                             initializer=tf.zeros_initializer)
        gamma = get_variable('gamma',
                              params_shape,
                              initializer=tf.ones_initializer)

        moving_mean = get_variable('moving_mean',
                                    params_shape,
                                    initializer=tf.zeros_initializer,
                                    trainable=False)
        moving_variance = get_variable('moving_variance',
                                        params_shape,
                                        initializer=tf.ones_initializer,
                                        trainable=False)

        # These ops will only be preformed when training.
        mean, variance = tf.nn.moments(x, axis)
        BN_DECAY=0.9997
        update_moving_mean = moving_averages.assign_moving_average(moving_mean,
                                                                   mean, BN_DECAY)
        update_moving_variance = moving_averages.assign_moving_average(
            moving_variance, variance, BN_DECAY)
        UPDATE_OPS_COLLECTION='resnet_update_ops'                               # must be grouped with training op
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_mean)
        tf.add_to_collection(UPDATE_OPS_COLLECTION, update_moving_variance)

        mean, variance = control_flow_ops.cond(
            c['is_training'], lambda: (mean, variance),
            lambda: (moving_mean, moving_variance))
        BN_EPSILON=0.001
        x = tf.nn.batch_normalization(x, mean, variance, beta, gamma, BN_EPSILON)
        # x.set_shape(inputs.get_shape()) ??

    return x

def max_pool(x, ksize=3, stride=2):
    return tf.nn.max_pool(x,
                          ksize=[1, ksize, ksize, 1],
                          strides=[1, stride, stride, 1],
                          padding='SAME')

def fc(x, c):
    FC_WEIGHT_STDDEV=0.00004
    num_units_in = x.get_shape()[1]
    num_units_out = c['fc_units_out']
    weights_initializer = tf.truncated_normal_initializer(
        stddev=FC_WEIGHT_STDDEV)

    weights_google = get_variable('weights',
                            shape=[1,1,num_units_in, num_units_out],
                            initializer=weights_initializer,
                            weight_decay=FC_WEIGHT_STDDEV)
    weights=weights_google[0,0,:,:]
    biases = get_variable('biases',
                           shape=[num_units_out],
                           initializer=tf.zeros_initializer)
    x = tf.nn.xw_plus_b(x, weights, biases)
    return x

def stack(x, c):
    for n in range(c['num_blocks']):
        s = c['stack_stride'] if n == 0 else 1                          #每个layer的第一个block中进行down_sampling
        c['block_stride'] = s
        with tf.variable_scope('unit_%d' % (n + 1)):
            with tf.variable_scope('bottleneck_v1'):
                x = block(x, c)
    return x

def block(x, c):
    filters_in = x.get_shape()[-1]

    # Note: filters_out isn't how many filters are outputed.
    # That is the case when bottleneck=False but when bottleneck is
    # True, filters_internal*4 filters are outputted. filters_internal is how many filters
    # the 3x3 convs output internally.
    m = 4 if c['bottleneck'] else 1
    filters_out = m * c['block_filters_internal']

    shortcut = x  # branch 1

    c['conv_filters_out'] = c['block_filters_internal']

    if c['bottleneck']:
        with tf.variable_scope('conv1'):
            c['ksize'] = 1
            c['stride'] = c['block_stride']                         #todo:1*1的卷积核，步长还大于1，那不是信息直接没了？
            x = conv(x, c)
            x = bn(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('conv2'):                                #todo:stride为2在scale3时不对
            c['ksize'] = 3
            c['stride'] = 1
            x = conv(x, c)
            x = bn(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('conv3'):
            c['conv_filters_out'] = filters_out
            c['ksize'] = 1
            #assert c['stride'] == 1
            c['stride'] = 1
            x = conv(x, c)
            x = bn(x, c)
    else:
        with tf.variable_scope('A'):
            c['stride'] = c['block_stride']
            #assert c['ksize'] == 3
            c['ksize'] = 3
            x = conv(x, c)
            x = bn(x, c)
            x = tf.nn.relu(x)

        with tf.variable_scope('B'):
            c['conv_filters_out'] = filters_out
            #assert c['ksize'] == 3
            #assert c['stride'] == 1
            c['ksize'] = 3
            c['stride'] = 1
            x = conv(x, c)
            x = bn(x, c)

    with tf.variable_scope('shortcut'):
        if filters_out != filters_in or c['block_stride'] != 1:
            c['ksize'] = 1
            c['stride'] = c['block_stride']
            c['conv_filters_out'] = filters_out
            shortcut = conv(shortcut, c)
            shortcut = bn(shortcut, c)
    return tf.nn.relu(x + shortcut)

def get_variable(name,
                  shape,
                  initializer,
                  weight_decay=0.0,
                  dtype='float',
                  trainable=True):
    "A little wrapper around tf.get_variable to do weight decay and add to"
    "resnet collection"
    if weight_decay > 0:
        regularizer = tf.contrib.layers.l2_regularizer(weight_decay)
    else:
        regularizer = None
    collections = [tf.GraphKeys.VARIABLES, 'resnet_variables']
    return tf.get_variable(name,
                           shape=shape,
                           initializer=initializer,
                           dtype=dtype,
                           regularizer=regularizer,
                           collections=collections,
                           trainable=trainable)