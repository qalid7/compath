import tensorflow as tf


def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev, name='Gaussian_Init')
    return tf.Variable(initial, name=name)


def batch_norm(inputs, n_out, epsilon=0.001, scope=None, reuse=None):

    inputs_shape = inputs.get_shape()
    with tf.variable_scope(scope, 'BatchNorm', [inputs], reuse=reuse):
        axis = list(range(len(inputs_shape) - 1))
        # params_shape = inputs_shape[-1:]

        # beta, gamma = None, None

        beta = tf.Variable(tf.constant(0.0, shape=[n_out]), name='beta',
                           trainable=True)  # this value is added to the normalized tensor
        gamma = tf.Variable(tf.constant(1.0, shape=[n_out]), name='gamma',
                            trainable=True)  # this value is multiplied to the normalized tensor

        mean, variance = tf.nn.moments(inputs, axis)

        outputs = tf.nn.batch_normalization(inputs, mean, variance, beta, gamma, epsilon)
        outputs.set_shape(inputs.get_shape())
    return outputs


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape, name='Constant_Init')
    return tf.Variable(initial, name=name)


def convolution_2d_full(in_tensor, in_ch_dim, out_ch_dim, is_batch_norm, layer_name, kdim1=3, kdim2=3,
                        strides=None, padding='VALID'):
    if strides is None:
        strides = [1, 1, 1, 1]

    kshape = [kdim1, kdim2, in_ch_dim, out_ch_dim]
    stddev = tf.sqrt(2 / (kshape[0] * kshape[1] * kshape[2]))

    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(kshape, stddev=stddev, name=layer_name + '_Weights')
            # variable_summaries(weights, 'weights_' + layer_name)
            convolution = tf.nn.conv2d(in_tensor, weights, strides=strides, padding=padding)
            tf.add_to_collection('losses', tf.nn.l2_loss(weights))
        if is_batch_norm:  # training and norm
            # outputs = layers.batch_norm(conv, scope=layer_name+'/BN')
            outputs = batch_norm(convolution, out_ch_dim)
        else:
            with tf.name_scope('biases'):
                biases = bias_variable([kshape[3]], name=layer_name + '_Bias')
                # variable_summaries(biases, 'biases_' + layer_name)
                outputs = tf.nn.bias_add(convolution, biases)
                tf.add_to_collection('losses', tf.nn.l2_loss(biases))

    return outputs


def deconvolution_2d_full(in_tensor, in_ch_dim, out_ch_dim, is_batch_norm, layer_name, output_shape, kdim1=2, kdim2=2,
                          strides=None, padding="SAME"):
    if strides is None:
        strides = [1, 2, 2, 1]
    kshape = [kdim1, kdim2, in_ch_dim, out_ch_dim]
    stddev = tf.sqrt(2 / (kshape[0] * kshape[1] * kshape[2]))
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            weights = weight_variable(kshape, stddev=stddev, name=layer_name + '_Weights')
            # variable_summaries(weights, 'weights_' + layer_name)
            deconv = tf.nn.conv2d_transpose(in_tensor, weights, output_shape, strides=strides, padding=padding)
            tf.add_to_collection('losses', tf.nn.l2_loss(weights))

        if is_batch_norm:
            # outputs = layers.batch_norm(deconv, scope=layer_name+'/BN')
            outputs = batch_norm(deconv, output_shape[3])
        else:
            with tf.name_scope('biases'):
                biases = bias_variable([kshape[2]], name=layer_name + '_Bias')
                # variable_summaries(biases, 'biases_' + layer_name)
                outputs = tf.nn.bias_add(deconv, biases)
                tf.add_to_collection('losses', tf.nn.l2_loss(biases))
    return outputs


def activation(x):
    return tf.nn.tanh(x)


def max_pool(x, ksize=None, strides=None, padding='SAME'):
    # 2x2 Max Pool
    if ksize is None:
        ksize = [1, 2, 2, 1]

    if strides is None:
        strides = [1, 2, 2, 1]

    return tf.nn.max_pool(x, ksize=ksize, strides=strides, padding=padding)


def softmax_classification(mask, name):
    name = name + '_SoftMax'
    with tf.name_scope(name):
        exp_mask = tf.exp(mask)
        sum_mask = tf.reduce_sum(exp_mask, 3, keep_dims=True)
        soft_mask = tf.div(exp_mask, sum_mask)
        return soft_mask


def inference(network, is_training, images=None):

    batch_size = network.batch_size
    in_feat_dim = network.in_feat_dim
    num_of_classes = network.num_of_classes

    images = tf.map_fn(lambda img: tf.image.per_image_standardization(img), images)

    out_conv_dim = 64
    name = 'Branch1'
    new_img_size = [256, 256]
    with tf.name_scope(name):
        conv1_1 = convolution_2d_full(images, in_feat_dim, out_conv_dim, True, name + '_Convolution_1',
                                      kdim1=3, kdim2=3)
        r_conv1_1 = activation(conv1_1)
        conv1_2 = convolution_2d_full(r_conv1_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv1_2 = activation(conv1_2)
        maxpool1_1 = max_pool(r_conv1_2)
        imrsize1_1 = tf.image.resize_bicubic(images, new_img_size)
        conv1_3 = convolution_2d_full(imrsize1_1, in_feat_dim, out_conv_dim, True, name + '_Convolution_3',
                                      kdim1=3, kdim2=3)
        r_conv1_3 = activation(conv1_3)
        conv1_4 = convolution_2d_full(r_conv1_3, out_conv_dim, out_conv_dim, False, name + '_Convolution_4',
                                      kdim1=3, kdim2=3)
        r_conv1_4 = activation(conv1_4)
        b1 = tf.concat(values=[maxpool1_1, r_conv1_4], axis=3)

    in_ch_dim = out_conv_dim*2
    out_conv_dim = 128
    name = 'Branch2'
    new_img_size = [128, 128]
    with tf.name_scope(name):
        conv2_1 = convolution_2d_full(b1, in_ch_dim, out_conv_dim, False, name + '_Convolution_1', kdim1=3, kdim2=3)
        r_conv2_1 = activation(conv2_1)
        conv2_2 = convolution_2d_full(r_conv2_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv2_2 = activation(conv2_2)
        maxpool2_1 = max_pool(r_conv2_2)
        imrsize2_1 = tf.image.resize_bicubic(images, new_img_size)
        conv2_3 = convolution_2d_full(imrsize2_1, in_feat_dim, out_conv_dim, True, name + '_Convolution_3',
                                      kdim1=3, kdim2=3)
        r_conv2_3 = activation(conv2_3)
        conv2_4 = convolution_2d_full(r_conv2_3, out_conv_dim, out_conv_dim, False, name + '_Convolution_4',
                                      kdim1=3, kdim2=3)
        r_conv2_4 = activation(conv2_4)
        b2 = tf.concat(values=[maxpool2_1, r_conv2_4], axis=3)

    in_ch_dim = out_conv_dim * 2
    out_conv_dim = 256
    name = 'Branch3'
    new_img_size = [64, 64]
    with tf.name_scope(name):
        conv3_1 = convolution_2d_full(b2, in_ch_dim, out_conv_dim, False, name + '_Convolution_1', kdim1=3, kdim2=3)
        r_conv3_1 = activation(conv3_1)
        conv3_2 = convolution_2d_full(r_conv3_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv3_2 = activation(conv3_2)
        maxpool3_1 = max_pool(r_conv3_2)
        imrsize3_1 = tf.image.resize_bicubic(images, new_img_size)
        conv3_3 = convolution_2d_full(imrsize3_1, in_feat_dim, out_conv_dim, True, name + '_Convolution_3',
                                      kdim1=3, kdim2=3)
        r_conv3_3 = activation(conv3_3)
        conv3_4 = convolution_2d_full(r_conv3_3, out_conv_dim, out_conv_dim, False, name + '_Convolution_4',
                                      kdim1=3, kdim2=3)
        r_conv3_4 = activation(conv3_4)
        b3 = tf.concat(values=[maxpool3_1, r_conv3_4], axis=3)

    in_ch_dim = out_conv_dim * 2
    out_conv_dim = 512
    name = 'Branch4'
    new_img_size = [32, 32]
    with tf.name_scope(name):
        conv4_1 = convolution_2d_full(b3, in_ch_dim, out_conv_dim, False, name + '_Convolution_1', kdim1=3, kdim2=3)
        r_conv4_1 = activation(conv4_1)
        conv4_2 = convolution_2d_full(r_conv4_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv4_2 = activation(conv4_2)
        maxpool4_1 = max_pool(r_conv4_2)
        imrsize4_1 = tf.image.resize_bicubic(images, new_img_size)
        conv4_3 = convolution_2d_full(imrsize4_1, in_feat_dim, out_conv_dim, True, name + '_Convolution_3',
                                      kdim1=3, kdim2=3)
        r_conv4_3 = activation(conv4_3)
        conv4_4 = convolution_2d_full(r_conv4_3, out_conv_dim, out_conv_dim, False, name + '_Convolution_4',
                                      kdim1=3, kdim2=3)
        r_conv4_4 = activation(conv4_4)
        b4 = tf.concat(values=[maxpool4_1, r_conv4_4], axis=3)

    in_ch_dim = out_conv_dim*2
    out_conv_dim = 1024*2
    name = 'Branch5'
    with tf.name_scope(name):
        conv5_1 = convolution_2d_full(b4, in_ch_dim, out_conv_dim, False, name + '_Convolution_1',
                                      kdim1=3, kdim2=3)
        r_conv5_1 = activation(conv5_1)
        conv5_2 = convolution_2d_full(r_conv5_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv5_2 = activation(conv5_2)
        maxpool5_1 = max_pool(r_conv5_2)
        conv5_3 = convolution_2d_full(maxpool5_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_3',
                                      kdim1=3, kdim2=3)
        r_conv5_3 = activation(conv5_3)
        conv5_3 = convolution_2d_full(r_conv5_3, out_conv_dim, out_conv_dim, False, name + '_Convolution_4',
                                      kdim1=3, kdim2=3)
        b5 = activation(conv5_3)

    in_ch_dim = out_conv_dim
    name = 'Branch5up'
    outputshape = [batch_size, int(b5.get_shape()[1].value) * 2, int(b5.get_shape()[2].value) * 2, out_conv_dim]
    with tf.name_scope(name):
        upconv5up_1 = deconvolution_2d_full(b5, out_conv_dim, in_ch_dim, False, name + '_Deconvolution_1', outputshape)
        conv5up_1 = convolution_2d_full(upconv5up_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_1',
                                        kdim1=3, kdim2=3)
        r_conv5up_1 = activation(conv5up_1)
        conv5up_2 = convolution_2d_full(r_conv5up_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                        kdim1=3, kdim2=3)
        r_conv5up_2 = activation(conv5up_2)
        b5up = deconvolution_2d_full(r_conv5up_2, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_2',
                                     outputshape, kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")

    in_ch_dim = out_conv_dim
    out_conv_dim = int(out_conv_dim / 2)
    name = 'Branch6'
    outputshape = [batch_size, int(b5up.get_shape()[1].value) * 2, int(b5up.get_shape()[2].value) * 2, out_conv_dim]
    # paddings = [[0, 0], [2, 2], [2, 2], [0, 0]]
    with tf.name_scope(name):
        upconv6_1 = deconvolution_2d_full(b5up, out_conv_dim, in_ch_dim, False, name + '_Deconvolution_1', outputshape)
        conv6_1 = convolution_2d_full(upconv6_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_1',
                                      kdim1=3, kdim2=3)
        r_conv6_1 = activation(conv6_1)
        conv6_2 = convolution_2d_full(r_conv6_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv6_2 = activation(conv6_2)
        upconv6_2 = deconvolution_2d_full(r_conv6_2, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_2',
                                          outputshape, kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        upconv6_3 = deconvolution_2d_full(b4, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_3', outputshape,
                                          kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        b6cat = tf.concat(values=[upconv6_3, upconv6_2], axis=3)
        b6conv = convolution_2d_full(b6cat, out_conv_dim + out_conv_dim, out_conv_dim, False, name + '_Convolution_3',
                                     kdim1=1, kdim2=1)
        b6 = activation(b6conv)

    in_ch_dim = out_conv_dim
    out_conv_dim = int(out_conv_dim / 2)
    name = 'Branch7'
    outputshape = [batch_size, int(b6.get_shape()[1].value) * 2, int(b6.get_shape()[2].value) * 2, out_conv_dim]
    # paddings = [[0, 0], [2, 2], [2, 2], [0, 0]]
    with tf.name_scope(name):
        upconv7_1 = deconvolution_2d_full(b6, out_conv_dim, in_ch_dim, False, name + '_Deconvolution_1', outputshape)
        conv7_1 = convolution_2d_full(upconv7_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_1',
                                      kdim1=3, kdim2=3)
        r_conv7_1 = activation(conv7_1)
        conv7_2 = convolution_2d_full(r_conv7_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv7_2 = activation(conv7_2)
        upconv7_2 = deconvolution_2d_full(r_conv7_2, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_2',
                                          outputshape, kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        upconv7_3 = deconvolution_2d_full(b3, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_3', outputshape,
                                          kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        b7cat = tf.concat(values=[upconv7_3, upconv7_2], axis=3)
        b7conv = convolution_2d_full(b7cat, out_conv_dim + out_conv_dim, out_conv_dim, False, name + '_Convolution_3',
                                     kdim1=1, kdim2=1)
        b7 = activation(b7conv)

    in_ch_dim = out_conv_dim
    out_conv_dim = int(out_conv_dim / 2)
    name = 'Branch8'
    outputshape = [batch_size, int(b7.get_shape()[1].value) * 2, int(b7.get_shape()[2].value) * 2, out_conv_dim]
    # paddings = [[0, 0], [2, 2], [2, 2], [0, 0]]
    with tf.name_scope(name):
        upconv8_1 = deconvolution_2d_full(b7, out_conv_dim, in_ch_dim, False, name + '_Deconvolution', outputshape)
        conv8_1 = convolution_2d_full(upconv8_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_1',
                                      kdim1=3, kdim2=3)
        r_conv8_1 = activation(conv8_1)
        conv8_2 = convolution_2d_full(r_conv8_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv8_2 = activation(conv8_2)
        upconv8_2 = deconvolution_2d_full(r_conv8_2, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_2',
                                          outputshape, kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        upconv8_3 = deconvolution_2d_full(b2, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_3', outputshape,
                                          kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        b8cat = tf.concat(values=[upconv8_3, upconv8_2], axis=3)
        b8conv = convolution_2d_full(b8cat, out_conv_dim + out_conv_dim, out_conv_dim, False, name + '_Convolution_3',
                                     kdim1=1, kdim2=1)
        b8 = activation(b8conv)

    in_ch_dim = out_conv_dim
    out_conv_dim = int(out_conv_dim / 2)
    name = 'Branch9'
    outputshape = [batch_size, int(b8.get_shape()[1].value) * 2, int(b8.get_shape()[2].value) * 2, out_conv_dim]
    # paddings = [[0, 0], [2, 2], [2, 2], [0, 0]]
    with tf.name_scope(name):
        upconv9_1 = deconvolution_2d_full(b8, out_conv_dim, in_ch_dim, False, name + '_Deconvolution', outputshape)
        conv9_1 = convolution_2d_full(upconv9_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_1',
                                      kdim1=3, kdim2=3)
        r_conv9_1 = activation(conv9_1)
        conv9_2 = convolution_2d_full(r_conv9_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
                                      kdim1=3, kdim2=3)
        r_conv9_2 = activation(conv9_2)
        upconv9_2 = deconvolution_2d_full(r_conv9_2, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_2',
                                          outputshape, kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        upconv9_3 = deconvolution_2d_full(b1, out_conv_dim, out_conv_dim, False, name + '_Deconvolution_3', outputshape,
                                          kdim1=5, kdim2=5, strides=[1, 1, 1, 1], padding="VALID")
        b9cat = tf.concat(values=[upconv9_3, upconv9_2], axis=3)
        b9conv = convolution_2d_full(b9cat, out_conv_dim + out_conv_dim, out_conv_dim, False, name + '_Convolution_3',
                                     kdim1=1, kdim2=1)
        b9 = activation(b9conv)

    in_ch_dim = out_conv_dim
    out_conv_dim = int(out_conv_dim / 2)
    name = 'outputB3'
    outputshape = [batch_size, int(b9.get_shape()[1].value) * 2, int(b9.get_shape()[2].value) * 2, out_conv_dim]
    drop_out = 0.5
    with tf.name_scope(name):
        upconv_b3_1 = deconvolution_2d_full(b9, out_conv_dim, in_ch_dim, False, name + '_Deconvolution', outputshape)
        conv_b3_1 = convolution_2d_full(upconv_b3_1, out_conv_dim, num_of_classes, False, name + '_Convolution_1',
                                        kdim1=3, kdim2=3)
        rconv_b3_1 = activation(conv_b3_1)
        # convB3_2 = conv2d_full(r_convB3_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
        # kdim1=3, kdim2=3)
        # r_convB3_2 = activation(convB3_2)
        # convB3_3 = conv2d_full(r_convB3_2, out_conv_dim, numclasses, False, name + '_Convolution_3', kdim1=1, kdim2=1)
        # r_convB3_3 = activation(convB3_3)
        if drop_out < 1 and is_training is not None:
            mask_b3 = tf.nn.dropout(rconv_b3_1, drop_out)
        else:
            mask_b3 = rconv_b3_1
        output_b3 = convolution_2d_full(mask_b3, num_of_classes, num_of_classes, False, name + '_MaskConvolution_1',
                                        kdim1=3, kdim2=3)
        logits_b3 = softmax_classification(output_b3, 'OutputB3')

    in_ch_dim = int(b8.get_shape()[3].value)
    out_conv_dim = int(in_ch_dim / 2)
    name = 'outputB2'
    outputshape = [batch_size, int(b8.get_shape()[1].value) * 4, int(b8.get_shape()[2].value) * 4, out_conv_dim]
    drop_out = 0.5
    with tf.name_scope(name):
        upconv_b2_1 = deconvolution_2d_full(b8, out_conv_dim, in_ch_dim, False, name + '_Deconvolution', outputshape,
                                            kdim1=4, kdim2=4, strides=[1, 4, 4, 1])
        conv_b2_1 = convolution_2d_full(upconv_b2_1, out_conv_dim, num_of_classes, False, name + '_Convolution_1',
                                        kdim1=3, kdim2=3)
        rconv_b2_1 = activation(conv_b2_1)
        # convB2_2 = conv2d_full(r_convB2_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
        # kdim1=3, kdim2=3)
        # r_convB2_2 = activation(convB2_2)
        # convB2_3 = conv2d_full(r_convB2_2, out_conv_dim, numclasses, False, name + '_Convolution_3', kdim1=1, kdim2=1)
        # r_convB2_3 = activation(convB2_3)
        if drop_out < 1 and is_training is not None:
            mask_b2 = tf.nn.dropout(rconv_b2_1, drop_out)
        else:
            mask_b2 = rconv_b2_1
        output_b2 = convolution_2d_full(mask_b2, num_of_classes, num_of_classes, False, name + '_MaskConvolution_1',
                                        kdim1=3, kdim2=3)
        logits_b2 = softmax_classification(output_b2, 'OutputB2')

    in_ch_dim = int(b7.get_shape()[3].value)
    out_conv_dim = int(in_ch_dim / 2)
    name = 'outputB1'
    outputshape = [batch_size, int(b7.get_shape()[1].value) * 8, int(b7.get_shape()[2].value) * 8, out_conv_dim]
    drop_out = 0.5
    with tf.name_scope(name):
        upconv_b1_1 = deconvolution_2d_full(b7, out_conv_dim, in_ch_dim, False, name + '_Deconvolution', outputshape,
                                            kdim1=8, kdim2=8, strides=[1, 8, 8, 1])
        conv_b1_1 = convolution_2d_full(upconv_b1_1, out_conv_dim, num_of_classes, False, name + '_Convolution_1',
                                        kdim1=3, kdim2=3)
        rconv_b1_1 = activation(conv_b1_1)
        # convB1_2 = conv2d_full(r_convB1_1, out_conv_dim, out_conv_dim, False, name + '_Convolution_2',
        # kdim1=3, kdim2=3)
        # r_convB1_2 = activation(convB1_2)
        # convB1_3 = conv2d_full(r_convB1_2, out_conv_dim, numclasses, False, name + '_Convolution_3', kdim1=1, kdim2=1)
        # r_convB1_3 = activation(convB1_3)
        if drop_out < 1 and is_training is not None:
            mask_b1 = tf.nn.dropout(rconv_b1_1, drop_out)
        else:
            mask_b1 = rconv_b1_1
        output_b1 = convolution_2d_full(mask_b1, num_of_classes, num_of_classes, False, name + '_MaskConvolution_1',
                                        kdim1=3, kdim2=3)
        logits_b1 = softmax_classification(output_b1, 'OutputB1')

    name = 'Output'
    with tf.name_scope(name):
        output = tf.concat(values=[rconv_b1_1, rconv_b2_1, rconv_b3_1], axis=3, name=name + 'CatAuxOutput')
        if drop_out < 1 and is_training is not None:
            output = tf.nn.dropout(output, drop_out)
        else:
            output = output
        mask = convolution_2d_full(output, num_of_classes * 3, num_of_classes, False, name + '_Convolution',
                                   kdim1=3, kdim2=3)
        logits = softmax_classification(mask, 'OutputB1')

    return logits, logits_b1, logits_b2, logits_b3
