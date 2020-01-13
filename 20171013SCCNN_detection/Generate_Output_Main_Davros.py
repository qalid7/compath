import os
import sys
import sccnn_detection as sccnn
from subpackages import NetworkOptions

data_dir = sys.argv[1]
sub_dir_name = sys.argv[2]

d = {'tissue_segment_dir': '',
     'preprocessed_dir': None,
     'exp_dir': 'ExpDir'}
with open(os.path.join(data_dir, 'parameters-detection.txt')) as param:
    for line in param:
        a = line.split(' ')
        d[a[0]] = a[1].strip('\n')

print('results_dir: ' + d['results_dir'], flush=True)
print('tissue_segment_dir: ' + d['tissue_segment_dir'], flush=True)
print('file_name_pattern: ' + d['file_name_pattern'], flush=True)
print('date: ' + d['date'], flush=True)
print('exp_dir: ' + d['exp_dir'], flush=True)

opts = NetworkOptions.NetworkOptions(exp_dir=d['exp_dir'],
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=31,
                                     image_width=31,
                                     in_feat_dim=int(d['in_feat_dim']),
                                     label_height=13,
                                     label_width=13,
                                     in_label_dim=1,
                                     batch_size=90,
                                     data_dir=data_dir,
                                     results_dir=d['results_dir'],
                                     tissue_segment_dir=d['tissue_segment_dir'],
                                     current_epoch_num=0,
                                     file_name_pattern=d['file_name_pattern'],
                                     preprocessed_dir=d['preprocessed_dir'],
                                     pre_process=True)

opts.results_dir = os.path.join(opts.results_dir, d['date'])
if d['preprocessed_dir'] is None:
    opts.preprocessed_dir = os.path.join(opts.preprocessed_dir, d['date'])

if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir, exist_ok=True)
if not os.path.isdir(os.path.join(opts.results_dir, 'h5')):
    os.makedirs(os.path.join(opts.results_dir, 'h5'), exist_ok=True)
if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'), exist_ok=True)
if not os.path.isdir(os.path.join(opts.results_dir, 'csv')):
    os.makedirs(os.path.join(opts.results_dir, 'csv'), exist_ok=True)
if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed')):
    os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed'), exist_ok=True)

Network = sccnn.SCCNN(batch_size=opts.batch_size,
                      image_height=opts.image_height,
                      image_width=opts.image_width,
                      in_feat_dim=opts.in_feat_dim,
                      out_height=opts.label_height,
                      out_width=opts.label_width,
                      out_feat_dim=opts.in_label_dim,
                      radius=10)

print('opts.data_dir:' + os.path.join(opts.data_dir, sub_dir_name), flush=True)
print('opts.results_dir:' + os.path.join(opts.results_dir, sub_dir_name), flush=True)
print('opts.preprocessed_dir:' + os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name), flush=True)
print('opts.tissue_segmentation:' + os.path.join(opts.tissue_segment_dir, sub_dir_name), flush=True)
print('opts.file_name_pattern:' + opts.file_name_pattern, flush=True)
print('opts.pre_process:' + str(opts.pre_process), flush=True)
print('opts.exp_dir:' + opts.exp_dir, flush=True)
print('opts.checkpoint_dir:' + opts.checkpoint_dir, flush=True)

Network = Network.generate_output_sub_dir(opts=opts, sub_dir_name=sub_dir_name,
                                          save_pre_process=True,
                                          network_output=True,
                                          post_process=True)
