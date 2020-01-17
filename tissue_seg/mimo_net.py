import tensorflow as tf

from subpackages import random_crop
from subpackages import inference
from subpackages import loss_function
from subpackages import run_training
from subpackages import generate_output


class MIMONet:
    def __init__(self, batch_size, image_height, image_width, in_feat_dim,
                 label_height, label_width, in_label_dim,
                 num_of_classes=2, crop_height=None, crop_width=None, tf_device=None):
        if crop_height is None:
            crop_height = 508
        if crop_width is None:
            crop_width = 508
        if tf_device is None:
            tf_device = ['/gpu:0']

        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.in_feat_dim = in_feat_dim
        self.num_of_classes = num_of_classes
        self.in_label_dim = in_label_dim
        self.loss = None
        self.accuracy = None
        self.logits = None
        self.logits_b1 = None
        self.logits_b2 = None
        self.logits_b3 = None
        self.tf_device = tf_device
        self.LearningRate = None
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.label_height = label_height
        self.label_width = label_width
        # for d in self.tf_device:
        #     with tf.device(d):
        self.images = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.image_height,
                                            self.image_width, self.in_feat_dim])
        self.labels = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.label_height,
                                            self.label_width, self.in_label_dim])

    def run_checks(self, opts):
        assert (opts.image_height == self.image_height)
        assert (opts.image_width == self.image_width)
        assert (opts.in_feat_dim == self.in_feat_dim)
        assert (opts.label_dim == self.in_label_dim)
        assert (opts.num_of_classes == self.num_of_classes)
        return 0

    def random_crop(self, images=None, labels=None):
        if images is None:
            images = self.images

        if labels is None:
            labels = images.labels

        images, labels = random_crop.random_crop(network=self, images=images, labels=labels)

        return images, labels

    def inference(self, is_training, images=None):
        if images is None:
            images = self.images

        self.logits, self.logits_b1, self.logits_b2, self.logits_b3 = inference.inference(network=self,
                                                                                          is_training=is_training,
                                                                                          images=images)
        return self.logits, self.logits_b1, self.logits_b2, self.logits_b3

    def run_training(self, opts):
        network = run_training.run_training(network=self, opts=opts)

        return network

    def generate_output(self, opts):

        generate_output.generate_output(network=self, opts=opts)

    def generate_output_sub_dir(self, opts, sub_dir_name):
        output_path = generate_output.generate_output_sub_dir(network=self, opts=opts,
                                                              sub_dir_name=sub_dir_name)
        print('Output Files saved at:' + output_path)

    def loss_function(self, global_step=None, labels=None):
        if labels is None:
            labels = self.labels

        self.loss = loss_function.aux_plus_main_loss(network=self, labels=labels, global_step=global_step)

        return self.loss

    def train(self):
        loss = self.loss
        lr = self.LearningRate
        with tf.name_scope('Optimization'):
            train_op = tf.train.AdagradOptimizer(learning_rate=lr).minimize(loss)

        return train_op
