import tensorflow as tf
import glob
import os
import matlab.engine
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
import pandas as pd
import math

from subpackages import Patches


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'csv', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'csv', sub_dir_name))

    if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images')):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images'))


def pre_process_images(opts, sub_dir_name, eng=None):
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.eval('run initialize_matlab_variables.m', nargout=0)

    make_sub_dirs(opts, sub_dir_name)
    if opts.pre_process:
        matlab_input = {'input_path': os.path.join(opts.data_dir, sub_dir_name),
                        'feat': ['rgb'],
                        'output_path': opts.results_dir,
                        'sub_dir_name': sub_dir_name}
        eng.workspace['matlab_input'] = matlab_input
        eng.eval('run Pre_process_images.m', nargout=0)


def generate_network_output(opts, sub_dir_name, network, sess,
                            logits, eng=None):
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.eval('run initialize_matlab_variables.m', nargout=0)

    make_sub_dirs(opts, sub_dir_name)
    image_path = os.path.join(opts.data_dir, sub_dir_name)
    files = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Ss1.jpg')))
    for i in range(len(files)):
        print(files[i])
        image_path_full = os.path.join(opts.data_dir, sub_dir_name, files[i])
        if opts.pre_process:
            workspace = sio.loadmat(os.path.join(opts.results_dir, 'pre_processed', sub_dir_name,
                                                 os.path.basename(files[i][:-3]) + 'mat'))
            matlab_output = workspace['matlab_output']
            feat = np.array(matlab_output['feat'][0][0])
        else:
            feat = image_path_full

        patch_obj = Patches.Patches(
            img_patch_h=opts.image_height, img_patch_w=opts.image_width,
            stride_h=opts.stride_h, stride_w=opts.stride_w,
            label_patch_h=opts.label_height, label_patch_w=opts.label_width)

        image_patches = patch_obj.extract_patches(feat)
        opts.num_examples_per_epoch_for_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
            image_patches.shape
        label_patches = np.zeros([opts.num_examples_per_epoch_for_train, opts.label_height,
                                  opts.label_width, opts.num_of_classes], dtype=np.float32)
        train_count = int((opts.num_examples_per_epoch_for_train / opts.batch_size) + 1)

        start = 0
        start_time = time.time()
        for step in range(train_count):
            end = start + opts.batch_size
            data_train = image_patches[start:end, :, :, :]
            data_train = data_train.astype(np.float32, copy=False)
            data_train_float32 = data_train / 255.0
            logits_out = sess.run(
                logits,
                feed_dict={network.images: data_train_float32,
                           })
            label_patches[start:end] = logits_out

            if end + opts.batch_size > opts.num_examples_per_epoch_for_train - 1:
                end = opts.num_examples_per_epoch_for_train - opts.batch_size

            start = end

        output = patch_obj.merge_patches(label_patches)
        mat = {'output': output}
        mat_file_name = os.path.basename(files[i][:-3]) + 'mat'
        sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, mat_file_name), mat)

        duration = time.time() - start_time
        format_str = (
            '%s: file %d/ %d, (%.2f sec/file)')
        print(format_str % (datetime.now(), i + 1, len(files), float(duration)))

        eng.workspace['results_dir'] = opts.results_dir
        eng.workspace['image_path'] = image_path
        eng.workspace['sub_dir_name'] = sub_dir_name
        eng.eval('Save_Detection_Output_p(results_dir, sub_dir_name, image_path)', nargout=0)

        workspace = sio.loadmat(os.path.join(opts.results_dir, 'mat', sub_dir_name,
                                             os.path.basename(files[i][:-3]) + 'mat'))
        mat = workspace['mat']
        bin_label = mat['BinLabel'][0][0]
        bin_label = bin_label.astype('bool')
        slide_h = bin_label.shape[0]
        slide_w = bin_label.shape[1]
        cws_h = 125
        cws_w = 125
        iter_tot = 0
        cws_file = []
        has_tissue = []
        for h in range(int(math.ceil((slide_h - cws_h) / cws_h + 1))):
            for w in range(int(math.ceil((slide_w - cws_w) / cws_w + 1))):
                start_h = h * cws_h
                end_h = (h * cws_h) + cws_h
                start_w = w * cws_w
                end_w = (w * cws_w) + cws_w
                if end_h > slide_h:
                    end_h = slide_h

                if end_w > slide_w:
                    end_w = slide_w

                cws_file.append('Da' + str(iter_tot))
                curr_bin_label = bin_label[start_h:end_h, start_w:end_w]
                has_tissue.append(curr_bin_label.any())
                if curr_bin_label.any():
                    mat = {'bin_label': curr_bin_label}
                    sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name,
                                             cws_file[iter_tot] + '.mat'), mat)

                iter_tot = iter_tot + 1

        data_dict = {'cws_file': cws_file,
                     'has_tissue': has_tissue}

        df = pd.DataFrame.from_dict(data_dict)
        df.to_csv(os.path.join(opts.results_dir, 'csv', sub_dir_name + '.csv'), index=False)


def generate_output(network, opts):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    logits, _, _, _ = network.inference(images=network.images, is_training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    eng = matlab.engine.start_matlab()
    eng.eval('run initialize_matlab_variables.m', nargout=0)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path)

        for cws_n in range(0, len(cws_sub_dir)):
            curr_cws_sub_dir = cws_sub_dir[cws_n]
            print(curr_cws_sub_dir)
            sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))

            pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network,
                                    sess=sess, logits=logits, eng=eng)

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name):
    logits, _, _, _ = network.inference(images=network.images, is_training=False)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    with tf.Session() as sess:

        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path)

        eng = matlab.engine.start_matlab()
        eng.addpath(eng.genpath('matlab/'), nargout=0)
        eng.eval('run initialize_matlab_variables.m', nargout=0)

        pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)
        generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network,
                                sess=sess, logits=logits, eng=eng)
    return opts.results_dir
