import os
import sys
from shutil import copyfile
from shutil import copytree
from shutil import rmtree
import scipy.io as sio
import pickle


import mimo_net
from subpackages import NetworkOptions

data_dir = sys.argv[1]
param_file = sys.argv[2]

d = {'exp_dir': 'ExpDir'}
with open(os.path.join(data_dir, param_file)) as param:
    for line in param:
        a = line.split(' ')
        d[a[0]] = a[1].strip('\n')

print('exp_dir: ' + d['exp_dir'], flush=True)
print('train_data_filename:' + d['train_data_filename'], flush=True)
print('valid_data_filename:' + d['valid_data_filename'], flush=True)

opts = NetworkOptions.NetworkOptions(exp_dir=d['exp_dir'],
                                     num_examples_per_epoch_train=1,
                                     num_examples_per_epoch_valid=1,
                                     image_height=600,
                                     image_width=600,
                                     label_height=600,
                                     label_width=600,
                                     crop_height=508,
                                     crop_width=508,
                                     in_feat_dim=3,
                                     in_label_dim=4,
                                     num_of_classes=2,
                                     batch_size=1,
                                     num_of_epoch=500,
                                     data_dir=data_dir,
                                     train_data_filename=d['train_data_filename'],
                                     valid_data_filename=d['valid_data_filename'],
                                     current_epoch_num=0)

if os.path.isdir(os.path.join(opts.exp_dir, 'code')):
    rmtree(os.path.join(opts.exp_dir, 'code'))
    os.makedirs(os.path.join(opts.exp_dir, 'code'))

if not os.path.isdir(opts.exp_dir):
    os.makedirs(opts.exp_dir)
    os.makedirs(opts.checkpoint_dir)
    os.makedirs(opts.log_train_dir)
    os.makedirs(os.path.join(opts.exp_dir, 'code'))


Network = mimo_net.MIMONet(batch_size=opts.batch_size,
                           image_height=opts.image_height,
                           image_width=opts.image_width,
                           in_feat_dim=opts.in_feat_dim,
                           in_label_dim=opts.in_label_dim,
                           num_of_classes=opts.num_of_classes,
                           label_height=opts.label_height,
                           label_width=opts.label_width,
                           crop_height=508,
                           crop_width=508,
                           tf_device=opts.tf_device)

copyfile('Train_Network_Main.py', os.path.join(opts.exp_dir, 'code', 'Train_Network_Main.py'))
copyfile('mimo_net.py', os.path.join(opts.exp_dir, 'code', 'mimo_net.py'))
copytree('subpackages', os.path.join(opts.exp_dir, 'code', 'subpackages'))
# copytree('matlab', os.path.join(opts.exp_dir, 'code', 'matlab'))
mat = {'opts': opts}
sio.savemat(os.path.join(opts.exp_dir, 'code', 'opts.mat'), mat)
pickle.dump(opts, open(os.path.join(opts.exp_dir, 'code', 'opts.p'), 'wb'))

Network = Network.run_training(opts=opts)
