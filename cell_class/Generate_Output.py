import os

import sccnn_classifier as sccnn_classifier
from subpackages import NetworkOptions

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opts = NetworkOptions.NetworkOptions(exp_dir='ExpDir-IHC/',
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=51,
                                     image_width=51,
                                     in_feat_dim=3,
                                     in_label_dim=1,
                                     num_of_classes=4,
                                     batch_size=500,
                                     data_dir='R:\\tracerx\\Melanoma\\Quad\\data\\cws',
                                     results_dir='D:/tmp/results_diagostics-ihc/classification',
                                     detection_results_path='R:\\tracerx\\Melanoma\\Quad\\results\\detection\\20171017',
                                     tissue_segment_dir='',
                                     preprocessed_dir=None,
                                     current_epoch_num=0,
                                     file_name_pattern='CB12*',
                                     pre_process=True,
                                     color_code_file='IHC_CD4_CD8_FoxP3.csv')

opts.results_dir = (os.path.join(opts.results_dir, '20171019'))
opts.preprocessed_dir = os.path.join(opts.preprocessed_dir, '20171019')

if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir)
if not os.path.isdir(os.path.join(opts.results_dir, 'mat')):
    os.makedirs(os.path.join(opts.results_dir, 'mat'))
if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))
if not os.path.isdir(os.path.join(opts.results_dir, 'csv')):
    os.makedirs(os.path.join(opts.results_dir, 'csv'))
if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed')):
    os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed'))

Network = sccnn_classifier.SccnnClassifier(batch_size=opts.batch_size,
                                           image_height=opts.image_height,
                                           image_width=opts.image_width,
                                           in_feat_dim=opts.in_feat_dim,
                                           in_label_dim=opts.in_label_dim,
                                           num_of_classes=opts.num_of_classes)
Network.generate_output(opts=opts)
