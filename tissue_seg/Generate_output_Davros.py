import os
import sys
from subpackages import NetworkOptions
import mimo_net


data_dir = sys.argv[1]
sub_dir_name = sys.argv[2]

d = {'exp_dir': 'ExpDir'}
with open(os.path.join(data_dir, 'parameters-tissuesegment.txt')) as param:
    for line in param:
        a = line.split(' ')
        d[a[0]] = a[1].strip('\n')

print('results_dir: ' + d['results_dir'], flush=True)
print('file_name_pattern: ' + d['file_name_pattern'], flush=True)
print('date: ' + d['date'], flush=True)
print('exp_dir: ' + d['exp_dir'], flush=True)


opts = NetworkOptions.NetworkOptions(exp_dir=d['exp_dir'],
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
                                     data_dir=data_dir,
                                     results_dir=d['results_dir'],
                                     current_epoch_num=0,
                                     file_name_pattern=d['file_name_pattern'],
                                     pre_process=True, #False to disable matlab
                                     )

opts.results_dir = (os.path.join(opts.results_dir, '20171031'))
if not os.path.isdir(opts.results_dir):
    os.makedirs(opts.results_dir, exist_ok=True)
    os.makedirs(os.path.join(opts.results_dir, 'mat'), exist_ok=True)
    os.makedirs(os.path.join(opts.results_dir, 'annotated_images'), exist_ok=True)
    os.makedirs(os.path.join(opts.results_dir, 'pre_processed'), exist_ok=True)
    os.makedirs(os.path.join(opts.results_dir, 'csv'), exist_ok=True)

Network = mimo_net.MIMONet(batch_size=opts.batch_size,
                           image_height=opts.image_height,
                           image_width=opts.image_width,
                           in_feat_dim=opts.in_feat_dim,
                           in_label_dim=opts.in_label_dim,
                           num_of_classes=opts.num_of_classes,
                           label_height=opts.label_height,
                           label_width=opts.label_width
                           )


print('opts.data_dir:' + os.path.join(opts.data_dir, sub_dir_name), flush=True)
print('opts.results_dir:' + os.path.join(opts.results_dir, sub_dir_name), flush=True)
print('opts.file_name_pattern:' + opts.file_name_pattern, flush=True)
print('opts.pre_process:' + str(opts.pre_process), flush=True)
print('opts.exp_dir:' + opts.exp_dir, flush=True)

Network.generate_output_sub_dir(opts=opts, sub_dir_name=sub_dir_name)
