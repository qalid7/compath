import tensorflow as tf
import matlab.engine
import numpy as np
import glob
import os
import time
from datetime import datetime

from subpackages import Patches
from subpackages import h5


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'h5', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'h5', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'annotated_images', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.results_dir, 'csv', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'csv', sub_dir_name))
    if not os.path.isdir(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name)):
        os.makedirs(os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name))


def pre_process_images(opts, sub_dir_name, eng=None):
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.eval('run initialize_matlab_variables.m', nargout=0)

    make_sub_dirs(opts, sub_dir_name)
    if opts.pre_process:
        matlab_input = {'input_path': os.path.join(opts.data_dir, sub_dir_name),
                        'feat': ['h', 'rgb'],
                        'output_path': opts.preprocessed_dir,
                        'sub_dir_name': sub_dir_name,
                        'tissue_segment_dir': opts.tissue_segment_dir}
        eng.pre_process_images(matlab_input, nargout=0)


def post_process_images(opts, sub_dir_name, eng=None):
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.eval('run initialize_matlab_variables.m', nargout=0)

    make_sub_dirs(opts, sub_dir_name)
    eng.save_detection_output_p(opts.results_dir, sub_dir_name, os.path.join(opts.data_dir, sub_dir_name),
                                nargout=0)


def generate_network_output(opts, sub_dir_name, network, sess, logits):
    make_sub_dirs(opts, sub_dir_name)
    if opts.tissue_segment_dir == '':
        files_tissue = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
    else:
        files_tissue = sorted(glob.glob(os.path.join(opts.tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat')))
    for i in range(len(files_tissue)):
        file_name = os.path.basename(files_tissue[i])
        file_name = file_name[:-4]
        if not os.path.isfile(os.path.join(opts.results_dir, 'h5', sub_dir_name, file_name + '.h5')):
            print(file_name, flush=True)
            image_path_full = os.path.join(opts.data_dir, sub_dir_name, file_name + '.jpg')
            if opts.pre_process:
                feat = h5.h5read(
                    filename=os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, file_name + '.h5'),
                    data_name='feat')
            else:
                feat = image_path_full

            patch_obj = Patches.Patches(
                img_patch_h=opts.image_height, img_patch_w=opts.image_width, stride_h=8, stride_w=8,
                label_patch_h=opts.label_height, label_patch_w=opts.label_width)

            image_patches = patch_obj.extract_patches(feat)
            opts.num_examples_per_epoch_for_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
                image_patches.shape
            label_patches = np.zeros([opts.num_examples_per_epoch_for_train, opts.label_height,
                                      opts.label_width, opts.in_label_dim], dtype=np.float32)
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
            h5_file_name = file_name + '.h5'
            h5.h5write(filename=os.path.join(opts.results_dir, 'h5', sub_dir_name, h5_file_name),
                       data_to_save=output,
                       name_data='output')

            duration = time.time() - start_time
            format_str = (
                '%s: file %d/ %d, (%.2f sec/file)')
            print(format_str % (datetime.now(), i + 1, len(files_tissue), float(duration)), flush=True)
        else:
            print('Already detected %s/%s\n' % (sub_dir_name, file_name), flush=True)


def generate_output(network, opts, save_pre_process=True, network_output=True, post_process=True):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    logits = network.inference()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)

        eng = matlab.engine.start_matlab()

        for cws_n in range(0, len(cws_sub_dir)):
            curr_cws_sub_dir = cws_sub_dir[cws_n]
            print(curr_cws_sub_dir, flush=True)
            sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))

            eng.eval('run initialize_matlab_variables.m', nargout=0)
            if save_pre_process:
                pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

            # files = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
            if network_output:
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess, logits=logits)

            if post_process:
                post_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
    logits = network.inference()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    with tf.Session() as sess:

        eng = matlab.engine.start_matlab()

        eng.eval('run initialize_matlab_variables.m', nargout=0)
        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

        if network_output:
            ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
            assert ckpt, "No Checkpoint file found"
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess, logits=logits)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

    return opts.results_dir
