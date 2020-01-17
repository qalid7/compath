import tensorflow as tf


def loss(logits, labels):
    with tf.name_scope('Cost_Function'):
        # CostFunction
        cross_entropy_unet = -tf.reduce_sum(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
        _ = tf.summary.scalar('Loss', cross_entropy_unet)
    return cross_entropy_unet


def weighed_loss(logits, label_train, name=None):
    labels = label_train[:, :, :, 0:2]
    weights = label_train[:, :, :, 2:4]
    name = name + 'Cost_Function'
    with tf.name_scope(name):
        # CostFunction
        epsilon = 1e-6
        clipped_logits = tf.clip_by_value(logits, epsilon, 1.0 - epsilon)
        log_loss = -labels * tf.log(clipped_logits)  - (1.0 - labels) * tf.log(1.0 - clipped_logits)
        cross_entropy_unet = tf.reduce_mean(tf.reduce_sum(log_loss * weights))
        _ = tf.summary.scalar('Loss_Weighed_' + name, cross_entropy_unet)
    return cross_entropy_unet


def aux_plus_main_loss(network, labels, global_step):
    logits_b1 = network.logits_b1
    logits_b2 = network.logits_b2
    logits_b3 = network.logits_b3
    logits = network.logits
    with tf.name_scope('Loss'):
        aux_loss = weighed_loss(logits_b1, labels, name='B1') + weighed_loss(logits_b2, labels, name='B2') + \
                  weighed_loss(logits_b3, labels, name='B3')
        main_loss = weighed_loss(logits, labels, name='Output')
        total_loss = main_loss + (aux_loss/(global_step+1))
        _ = tf.summary.scalar('Loss_Total', total_loss)
    return total_loss
