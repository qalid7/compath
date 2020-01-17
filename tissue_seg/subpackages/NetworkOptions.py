import os
import numpy as np


class NetworkOptions:

    def __init__(self,
                 exp_dir=os.path.join(os.getcwd(), 'ExpDir'),
                 num_examples_per_epoch_train=1,
                 num_examples_per_epoch_valid=1,
                 image_height=600,
                 image_width=600,
                 in_feat_dim=3,
                 in_label_dim=4,
                 label_height=600,
                 label_width=600,
                 crop_height=508,
                 crop_width=508,
                 batch_size=1,
                 num_of_epoch=500,
                 stride_h=400,
                 stride_w=400,
                 data_dir=os.getcwd(),
                 results_dir=os.getcwd(),
                 train_data_filename='TrainData.h5',
                 valid_data_filename='ValidData.h5',
                 current_epoch_num=0,
                 file_name_pattern='TMA_*',
                 num_of_classes=2,
                 pre_process=False,
                 tf_device=None
                 ):
        if tf_device is None:
            tf_device = ['/gpu:0']
        self.data_dir = os.path.normpath(data_dir)
        self.train_data_filename = train_data_filename
        self.valid_data_filename = valid_data_filename
        self.exp_dir = os.path.normpath(exp_dir)
        self.checkpoint_dir = os.path.join(self.exp_dir, 'checkpoint')
        self.log_train_dir = os.path.join(self.exp_dir, 'logs')
        self.results_dir = os.path.normpath(results_dir)
        self.num_examples_per_epoch_train = num_examples_per_epoch_train
        self.num_examples_per_epoch_valid = num_examples_per_epoch_valid
        self.image_height = image_height
        self.image_width = image_width
        self.in_feat_dim = in_feat_dim
        self.in_label_dim = in_label_dim
        self.num_of_epoch = num_of_epoch
        self.batch_size = batch_size
        self.current_epoch_num = current_epoch_num
        self.num_of_classes = num_of_classes
        self.file_name_pattern = file_name_pattern
        self.pre_process = pre_process
        self.tf_device = tf_device
        self.label_height = label_height
        self.label_width = label_width
        self.crop_height = crop_height
        self.crop_width = crop_width
        self.stride_h = stride_h
        self.stride_w = stride_w
