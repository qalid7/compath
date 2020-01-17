import os

import mimo_net
from subpackages import NetworkOptions

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opts = NetworkOptions.NetworkOptions(exp_dir='ExpDir/',
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=508,
                                     image_width=508,
                                     label_height=508,
                                     label_width=508,
                                     in_feat_dim=3,
                                     in_label_dim=4,
                                     num_of_classes=2,
                                     batch_size=1,
                                     data_dir='R:\\tracerx\\Melanoma\\Quad\\data\\cws',
                                     results_dir='R:\\tracerx\\Melanoma\\Quad\\results\\'
                                                 'tissue_segmentation',
                                     current_epoch_num=0,
                                     file_name_pattern='*.ndpi',
                                     pre_process=True,
                                     )

opts.results_dir = (os.path.join(opts.results_dir, '20171019'))
if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir)
    os.makedirs(os.path.join(opts.results_dir, 'mat'))
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))
    os.makedirs(os.path.join(opts.results_dir, 'pre_processed'))
    os.makedirs(os.path.join(opts.results_dir, 'csv'))

Network = mimo_net.MIMONet(batch_size=opts.batch_size,
                           image_height=opts.image_height,
                           image_width=opts.image_width,
                           in_feat_dim=opts.in_feat_dim,
                           in_label_dim=opts.in_label_dim,
                           num_of_classes=opts.num_of_classes,
                           label_height=opts.label_height,
                           label_width=opts.label_width
                           )
Network.generate_output(opts=opts)
