import tensorflow as tf


def random_crop(network, images, labels):
    concat = tf.concat(values=[images, labels], axis=3)
    cropped = tf.map_fn(lambda img: tf.random_crop(img, [network.crop_height, network.crop_width,
                                                         network.in_feat_dim+network.in_label_dim]), concat)
    images = cropped[:, :, :, 0:network.in_feat_dim]
    labels = cropped[:, :, :, network.in_feat_dim:network.in_feat_dim + network.in_label_dim]

    return images, labels
