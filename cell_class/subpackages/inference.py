import tensorflow as tf


def convolution_2d_full(in_tensor, in_ch_dim, out_ch_dim, not_bias, layer_name, k_dim):
    k_shape = [k_dim, k_dim, in_ch_dim, out_ch_dim]
    stddev = tf.sqrt(2 / (k_shape[0] * k_shape[1] * k_shape[2]))
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(k_shape, stddev=stddev, name=layer_name + '_Weights')
            convolution = tf.nn.conv2d(in_tensor, weights, strides=[1, 1, 1, 1], padding='VALID')
            tf.add_to_collection('losses', tf.nn.l2_loss(weights))
        if not_bias:  # training and norm
            outputs = convolution
        else:
            with tf.name_scope('biases'):
                biases = bias_variable([k_shape[3]], name=layer_name + '_Bias')
                outputs = tf.nn.bias_add(convolution, biases)
                tf.add_to_collection('losses', tf.nn.l2_loss(biases))
    return outputs


def soft_max(in_tensor, name):
    name = name + '_SoftMax'
    with tf.name_scope(name):
        exp_mask = tf.exp(in_tensor)
        sum_mask = tf.reduce_sum(exp_mask, 3, keep_dims=True)
        soft_mask = tf.div(exp_mask, sum_mask)
        return soft_mask


def activation(in_tensor):
    return tf.nn.relu(in_tensor)


def max_pool(in_tensor):
    return tf.nn.max_pool(in_tensor, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev, name='Gaussian_Init')
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name='Constant_Init')
    return tf.Variable(initial, name=name)


def batch_norm(in_tensor, n_out, epsilon=0.001, scope=None, reuse=None):
    inputs_shape = in_tensor.get_shape()
    with tf.variable_op_scope([in_tensor], scope, 'BatchNorm', reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        # Allocate parameters for the beta and gamma of the normalization.
        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta',
                           trainable=True)  # this value is added to the normalized tensor
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma',
                            trainable=True)  # this value is multiplied to the normalized tensor
        #  Calculate the moments based on the individual batch.
        mean, variance = tf.nn.moments(in_tensor, axis)
        outputs = tf.nn.batch_normalization(in_tensor, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(in_tensor.get_shape())

    return outputs


def inference(network,
              is_training):
    images = network.images
    in_feat_dim = network.in_feat_dim
    num_of_classes = network.num_of_classes

    layer_id = 'L1'

    with tf.name_scope('Convolution_1'):
        in_ch_dim = in_feat_dim
        out_ch_dim = 32
        convolution_1 = convolution_2d_full(
            images, in_ch_dim, out_ch_dim, False, layer_id + '_Convolution_1', 2)

    with tf.name_scope('ReLU_1'):
        r_convolution_1 = activation(convolution_1)

    with tf.name_scope('Pooling_1'):
        pool_1 = max_pool(r_convolution_1)

    layer_id = 'L2'
    with tf.name_scope('Convolution_2'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 64
        convolution_2 = convolution_2d_full(
            pool_1, in_ch_dim, out_ch_dim, False, layer_id + '_Convolution_2', 2)

    with tf.name_scope('ReLU_2'):
        r_convolution_2 = activation(convolution_2)

    with tf.name_scope('Pooling'):
        pool_2 = max_pool(r_convolution_2)

    layer_id = 'L3'
    with tf.name_scope('Convolution_3'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 128
        convolution_3 = convolution_2d_full(
            pool_2, in_ch_dim, out_ch_dim, False, layer_id + '_Convolution_3', 3)

    with tf.name_scope('ReLU_3'):
        r_convolution_3 = activation(convolution_3)

    with tf.name_scope('Pooling'):
        pool_3 = max_pool(r_convolution_3)

    layer_id = 'L4'
    with tf.name_scope('Convolution_4'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 1024
        convolution_4 = convolution_2d_full(
            pool_3, in_ch_dim, out_ch_dim, False, layer_id + '_Convolution_4', 5)

    with tf.name_scope('ReLU_4'):
        r_convolution_4 = activation(convolution_4)

    with tf.name_scope('Dropout'):
        if is_training:
            dropout_4 = tf.nn.dropout(r_convolution_4, 0.5)
        else:
            dropout_4 = r_convolution_4

    layer_id = 'L5'
    with tf.name_scope('Convolution_5'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 512
        convolution_5 = convolution_2d_full(
            dropout_4, in_ch_dim, out_ch_dim, False, layer_id + '_Convolution_5', 1)

    with tf.name_scope('ReLU_4'):
        r_convolution_5 = activation(convolution_5)

    with tf.name_scope('Dropout'):
        if is_training:
            dropout_5 = tf.nn.dropout(r_convolution_5, 0.5)
        else:
            dropout_5 = r_convolution_5

    layer_id = 'L6'
    with tf.name_scope('Convolution_6'):
        in_ch_dim = out_ch_dim
        out_ch_dim = num_of_classes
        convolution_6 = convolution_2d_full(
            dropout_5, in_ch_dim, out_ch_dim, False, layer_id + '_Convolution_6', 1)

    with tf.variable_scope('soft_max'):
        softmax = soft_max(convolution_6, layer_id)

    logits = softmax

    output = {'convolution_1': convolution_1,
              'convolution_2': convolution_2,
              'convolution_3': convolution_3,
              'convolution_4': convolution_4,
              'convolution_5': convolution_5,
              'convolution_6': convolution_6
              }

    return logits, output
