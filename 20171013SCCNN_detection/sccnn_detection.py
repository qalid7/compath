import tensorflow as tf

from subpackages import inference
from subpackages import generate_output


class SCCNN:

    def __init__(self, batch_size, image_height, image_width,
                 in_feat_dim, out_height, out_width, out_feat_dim, radius=5, device='/gpu:1'):
        self.batch_size = batch_size
        self.image_height = image_height
        self.image_width = image_width
        self.in_feat_dim = in_feat_dim
        self.out_height = out_height
        self.out_width = out_width
        self.out_feat_dim = out_feat_dim
        self.radius = radius

        x, y = tf.meshgrid(tf.range(0, out_height), tf.range(0, out_width))

        x = tf.expand_dims(x, axis=2)  # Make 3D vector
        y = tf.expand_dims(y, axis=2)

        x = tf.expand_dims(x, axis=0)  # Make 4D vector
        y = tf.expand_dims(y, axis=0)

        x = tf.tile(x, multiples=[batch_size, 1, 1, 1])  # Make 3D vector
        y = tf.tile(y, multiples=[batch_size, 1, 1, 1])

        self.X = tf.cast(x, dtype=tf.float32)
        self.Y = tf.cast(y, dtype=tf.float32)

        self.images = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.image_height, self.image_width, self.in_feat_dim])

        self.labels = tf.placeholder(tf.float32,
                                     shape=[self.batch_size, self.out_height, self.out_width, self.out_feat_dim])

        self.train_op = None
        self.loss = None
        self.logits = None
        self.device = device
        self.LearningRate = None

    def run_checks(self, opts):
        assert (opts.image_height == self.image_height)
        assert (opts.image_width == self.image_width)
        assert (opts.in_feat_dim == self.in_feat_dim)
        assert (opts.label_height == self.out_height)
        assert (opts.label_width == self.out_width)
        assert (opts.in_label_dim == self.out_feat_dim)
        return 0

    def generate_output(self, opts, save_pre_process=True, network_output=True, post_process=True):
        output_path = generate_output.generate_output(network=self, opts=opts,
                                                      save_pre_process=save_pre_process,
                                                      network_output=network_output,
                                                      post_process=post_process)
        print('Output Files saved at:' + output_path)

    def generate_output_sub_dir(self, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
        output_path = generate_output.generate_output_sub_dir(network=self, opts=opts,
                                                              sub_dir_name=sub_dir_name,
                                                              save_pre_process=save_pre_process,
                                                              network_output=network_output,
                                                              post_process=post_process)
        print('Output Files saved at:' + output_path)

    def inference(self):
        self.logits = inference.inference(network=self)
        return self.logits
