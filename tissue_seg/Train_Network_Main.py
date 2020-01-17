import os
from shutil import copyfile
from shutil import copytree
from shutil import rmtree
import scipy.io as sio
import pickle

import mimo_net
from subpackages import NetworkOptions


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

opts = NetworkOptions.NetworkOptions(exp_dir=os.path.normpath(os.path.join(os.getcwd(), 'ExpDir')),
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
                                     data_dir=os.path.normpath('D:/Shan/MyCodes/TracerX/TissueSegmentation/Data'),
                                     train_data_filename='TrainData171017.h5',
                                     valid_data_filename='ValidData171017.h5',
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
