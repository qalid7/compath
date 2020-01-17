import tensorflow as tf

from subpackages import inference
from subpackages import generate_output


class SccnnClassifier:
    def __init__(self, batch_size, image_height, image_width, in_feat_dim,
                 in_label_dim, num_of_classes, tf_device=None):
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
        self.tf_device = tf_device
        self.LearningRate = None
        # for d in self.tf_device:
        #     with tf.device(d):
        self.images = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.image_height,
                                            self.image_width, self.in_feat_dim])
        self.labels = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.in_label_dim])

    def run_checks(self, opts):
        assert (opts.image_height == self.image_height)
        assert (opts.image_width == self.image_width)
        assert (opts.in_feat_dim == self.in_feat_dim)
        assert (opts.in_label_dim == self.in_label_dim)
        return 0

    def generate_output(self, opts, save_pre_process=True, network_output=True, post_process=True):

        generate_output.generate_output(network=self, opts=opts,
                                        save_pre_process=save_pre_process,
                                        network_output=network_output,
                                        post_process=post_process)

    def generate_output_sub_dir(self, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
        output_path = generate_output.generate_output_sub_dir(network=self, opts=opts,
                                                              sub_dir_name=sub_dir_name,
                                                              save_pre_process=save_pre_process,
                                                              network_output=network_output,
                                                              post_process=post_process)
        print('Output Files saved at:' + output_path)

    def inference(self, is_training):

        self.logits, output = inference.inference(network=self,
                                                  is_training=is_training)

        return self.logits, output

