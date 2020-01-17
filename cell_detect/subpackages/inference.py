import tensorflow as tf


def convolution2d_full(in_tensor, in_ch_dim, out_ch_dim, notbias, layer_name, kdim):
    k_shape = [kdim, kdim, in_ch_dim, out_ch_dim]
    stddev = tf.sqrt(2 / (k_shape[0] * k_shape[1] * k_shape[2]))
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(k_shape, stddev=stddev, name=layer_name + '_Weights')
            convolution = tf.nn.conv2d(in_tensor, weights, strides=[1, 1, 1, 1], padding='VALID')
            tf.add_to_collection('losses', tf.nn.l2_loss(weights))
        if notbias:  # training and norm
            outputs = convolution
        else:
            with tf.name_scope('biases'):
                biases = bias_variable([k_shape[3]], name=layer_name + '_Bias')
                outputs = tf.nn.bias_add(convolution, biases)
                tf.add_to_collection('losses', tf.nn.l2_loss(biases))

    return outputs


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


def batch_norm(in_tensor, n_out, epsilon=0.001, trainable=True, scope=None, reuse=None):
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


def SC1layer(network, inTensor, in_ch_dim, out_ch_dim):
    S1 = convolution2d_full(inTensor, in_ch_dim, out_ch_dim, False, 'Sigmoid' + '_FConvolution_1', 1)
    S1_sigmoid = tf.sigmoid(S1, name='Sigmoid')
    S1_sigmoid0 = S1_sigmoid[:, :, :, 0:1] * (network.out_height - 1)
    S1_sigmoid1 = S1_sigmoid[:, :, :, 1:2] * (network.out_width - 1)
    S1_sigmoid2 = S1_sigmoid[:, :, :, 2:3]
    tf.summary.histogram('S1_sigmoid0', S1_sigmoid0)
    tf.summary.histogram('S1_sigmoid1', S1_sigmoid1)
    tf.summary.histogram('S1_sigmoid2', S1_sigmoid2)
    tf.summary.scalar('mean/' + 'S1_sigmoid2', tf.reduce_mean(S1_sigmoid2))
    tf.summary.scalar('max/' + 'S1_sigmoid2', tf.reduce_max(S1_sigmoid2))
    tf.summary.scalar('min/' + 'S1_sigmoid2', tf.reduce_min(S1_sigmoid2))
    return S1_sigmoid0, S1_sigmoid1, S1_sigmoid2


def SC2layer(network, SC1_0, SC1_1, SC1_2):
    xvr = (network.X - SC1_0) ** 2
    yvc = (network.Y - SC1_1) ** 2
    M = xvr + yvc
    ML4 = tf.cast(tf.less(M, network.radius), dtype=tf.float32)
    numerator = SC1_2
    denominator = (1 + (M / 2))
    D = numerator / denominator
    D = D * ML4
    tf.summary.scalar('mean/' + 'D', tf.reduce_mean(D))
    tf.summary.scalar('max/' + 'D', tf.reduce_max(D))
    tf.summary.scalar('min/' + 'D', tf.reduce_min(D))
    tf.summary.histogram('D', D)
    return D


def inference(network, images=None):
    if images is None:
        images = network.images

    LayerID = 'L1'
    with tf.name_scope('Convolution_1'):
        in_ch_dim = network.in_feat_dim
        out_ch_dim = 30
        conv1 = convolution2d_full(images, in_ch_dim, out_ch_dim, False, LayerID + '_Convolution_1', 2)

    with tf.name_scope('ReLU_1'):
        r_conv1 = activation(conv1)

    with tf.name_scope('Pooling_1'):
        pool_1 = max_pool(r_conv1)

    LayerID = 'L2'
    with tf.name_scope('Convolution_2'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 60
        conv2 = convolution2d_full(pool_1, in_ch_dim, out_ch_dim, False, LayerID + '_Convolution_2', 2)

    with tf.name_scope('ReLU_2'):
        r_conv2 = activation(conv2)

    with tf.name_scope('Pooling'):
        pool_2 = max_pool(r_conv2)

    LayerID = 'L3'
    with tf.name_scope('Convolution_2'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 90
        conv3 = convolution2d_full(pool_2, in_ch_dim, out_ch_dim, False, LayerID + '_Convolution_2', 3)

    with tf.name_scope('ReLU_2'):
        r_conv3 = activation(conv3)

    LayerID = 'L4'
    with tf.name_scope('FConvolution_1'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 1024
        fconv1 = convolution2d_full(r_conv3, in_ch_dim, out_ch_dim, False, LayerID + '_FConvolution_1', 5)
    with tf.name_scope('FReLU_1'):
        r_Fconv1 = activation(fconv1)

    with tf.name_scope('Dropout'):
        r_Fconv1Dout = tf.nn.dropout(r_Fconv1, 0.5)

    LayerID = 'L5'
    with tf.name_scope('FConvolution_2'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 512
        fconv2 = convolution2d_full(r_Fconv1Dout, in_ch_dim, out_ch_dim, False, LayerID + '_FConvolution_1', 1)
    with tf.name_scope('FReLU_1'):
        r_Fconv2 = activation(fconv2)

    with tf.name_scope('Dropout'):
        r_Fconv2Dout = tf.nn.dropout(r_Fconv2, 0.5)

    with tf.name_scope('SC1'):
        in_ch_dim = out_ch_dim
        out_ch_dim = 3
        SC1_0, SC1_1, SC1_2 = SC1layer(network, r_Fconv2Dout, in_ch_dim, out_ch_dim)

    with tf.name_scope('SC2'):
        SC2 = SC2layer(network, SC1_0, SC1_1, SC1_2)

    network.logits = SC2

    return network.logits