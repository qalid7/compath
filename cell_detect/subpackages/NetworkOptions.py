import os


class NetworkOptions:

    def __init__(self,
                 exp_dir,
                 num_examples_per_epoch_train=1,
                 num_examples_per_epoch_valid=1,
                 image_height=31,
                 image_width=31,
                 in_feat_dim=4,
                 label_height=13,
                 label_width=13,
                 in_label_dim=1,
                 batch_size=500,
                 num_of_epoch=1000,
                 data_dir=os.getcwd(),
                 results_dir=os.getcwd(),
                 preprocessed_dir=None,
                 tissue_segment_dir='',
                 train_data_filename='TrainData.h5',
                 valid_data_filename='ValidData.h5',
                 current_epoch_num=0,
                 file_name_pattern='TMA_*',
                 pre_process=False):
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
        self.label_height = label_height
        self.label_width = label_width
        self.in_label_dim = in_label_dim
        self.num_of_epoch = num_of_epoch
        self.batch_size = batch_size
        self.current_epoch_num = current_epoch_num
        self.file_name_pattern = file_name_pattern
        self.pre_process = pre_process
        if tissue_segment_dir != '':
            self.tissue_segment_dir = os.path.normpath(tissue_segment_dir)
        else:
            self.tissue_segment_dir = ''
        if preprocessed_dir is None:
            self.preprocessed_dir = self.results_dir
        else:
            self.preprocessed_dir = preprocessed_dir
