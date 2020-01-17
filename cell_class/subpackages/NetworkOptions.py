import os
import numpy as np


class NetworkOptions:

    def __init__(self,
                 exp_dir=os.path.join(os.getcwd(), 'ExpDir'),
                 num_examples_per_epoch_train=1,
                 num_examples_per_epoch_valid=1,
                 image_height=51,
                 image_width=51,
                 in_feat_dim=3,
                 in_label_dim=1,
                 batch_size=5,
                 num_of_epoch=1000,
                 data_dir=os.getcwd(),
                 results_dir=os.getcwd(),
                 detection_results_path=os.getcwd(),
                 train_data_filename='TrainData.h5',
                 valid_data_filename='ValidData.h5',
                 output_filename='Labels.h5',
                 tissue_segment_dir='',
                 preprocessed_dir=None,
                 current_epoch_num=0,
                 file_name_pattern='TMA_*',
                 num_of_classes=2,
                 pre_process=False,
                 tf_device=None,
                 weighted_loss_per_class=None,
                 color_code_file='color_code_file.csv'):
        if weighted_loss_per_class is None:
            weighted_loss_per_class = np.ones(num_of_classes)
        if tf_device is None:
            tf_device = ['/gpu:0']
        self.data_dir = os.path.normpath(data_dir)
        self.detection_results_path = os.path.normpath(detection_results_path)
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
        self.output_filename = output_filename
        self.num_of_classes = num_of_classes
        self.file_name_pattern = file_name_pattern
        self.pre_process = pre_process
        self.tf_device = tf_device
        self.weighted_loss_per_class = weighted_loss_per_class
        if tissue_segment_dir != '':
            self.tissue_segment_dir = os.path.normpath(tissue_segment_dir)
        else:
            self.tissue_segment_dir = ''
        self.color_code_file = color_code_file
        if preprocessed_dir is None:
            self.preprocessed_dir = self.results_dir
        else:
            self.preprocessed_dir = os.path.normpath(preprocessed_dir)
