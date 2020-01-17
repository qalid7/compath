import tensorflow as tf
import glob
import os
import numpy as np
import scipy.io as sio
import time
from datetime import datetime
import matlab.engine

from subpackages import Patches
from subpackages import h5


def make_sub_dirs(opts, sub_dir_name):
    if not os.path.isdir(os.path.join(opts.results_dir, 'mat', sub_dir_name)):
        os.makedirs(os.path.join(opts.results_dir, 'mat', sub_dir_name))
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
                        'feat': ['rgb'],
                        'output_path': opts.preprocessed_dir,
                        'sub_dir_name': sub_dir_name,
                        'tissue_segment_dir': opts.tissue_segment_dir}
        eng.pre_process_images(matlab_input, nargout=0)


def generate_network_output(opts, sub_dir_name, network, sess, logits_labels, csv_detection_results_dir):
    make_sub_dirs(opts, sub_dir_name)
    if opts.tissue_segment_dir == '':
        files_tissue = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Da*.jpg')))
    else:
        files_tissue = sorted(glob.glob(os.path.join(opts.tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat')))

    for i in range(len(files_tissue)):
        file_name = os.path.basename(files_tissue[i])
        file_name = file_name[:-4]
        if not os.path.isfile(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat')):
            print(file_name, flush=True)
            image_path_full = os.path.join(opts.data_dir, sub_dir_name, file_name + '.jpg')
            if opts.pre_process:
                feat = h5.h5read(
                    filename=os.path.join(opts.preprocessed_dir, 'pre_processed', sub_dir_name, file_name + '.h5'),
                    data_name='feat')
            else:
                feat = image_path_full

            patch_obj = Patches.Patches(patch_h=opts.image_height, patch_w=opts.image_width)

            image_patches, labels, cell_id = patch_obj.extract_patches(
                input_image=feat,
                input_csv=os.path.join(csv_detection_results_dir, file_name + '.csv'))
            opts.num_examples_per_epoch_train, opts.image_height, opts.image_width, opts.in_feat_dim = \
                image_patches.shape
            label_patches = np.zeros([opts.num_examples_per_epoch_train, opts.in_label_dim], dtype=np.float32)
            train_count = int((opts.num_examples_per_epoch_train / opts.batch_size) + 1)

            start = 0
            start_time = time.time()

            if image_patches.shape[0] != 0 and opts.batch_size > opts.num_examples_per_epoch_train:
                image_patches_temp = image_patches
                for rs_var in range(int((opts.batch_size / opts.num_examples_per_epoch_train))):
                    image_patches_temp = np.concatenate((image_patches_temp, image_patches), axis=0)

                image_patches = image_patches_temp

            opts.num_examples_per_epoch_train_temp = image_patches.shape[0]

            if image_patches.shape[0] != 0:
                label_patches = np.zeros([opts.num_examples_per_epoch_train_temp, opts.in_label_dim], dtype=np.float32)
                for step in range(train_count):
                    end = start + opts.batch_size
                    data_train = image_patches[start:end, :, :, :]
                    data_train = data_train.astype(np.float32, copy=False)
                    data_train_float32 = data_train / 255.0
                    logits_out = sess.run(
                        logits_labels,
                        feed_dict={network.images: data_train_float32,
                                   })
                    label_patches[start:end] = np.squeeze(logits_out, axis=1) + 1

                    if end + opts.batch_size > opts.num_examples_per_epoch_train_temp - 1:
                        end = opts.num_examples_per_epoch_train_temp - opts.batch_size

                    start = end

                label_patches = label_patches[0:opts.num_examples_per_epoch_train]
            duration = time.time() - start_time
            mat = {'output': label_patches,
                   'labels': labels,
                   'cell_ids': cell_id}
            sio.savemat(os.path.join(opts.results_dir, 'mat', sub_dir_name, file_name + '.mat'), mat)
            format_str = (
                '%s: file %d/ %d, (%.2f sec/file)')
            print(format_str % (datetime.now(), i + 1, len(files_tissue), float(duration)), flush=True)
        else:
            print('Already classified %s/%s\n' % (sub_dir_name, file_name), flush=True)


def post_process_images(opts, sub_dir_name, csv_detection_results_dir, eng=None):
    if eng is None:
        eng = matlab.engine.start_matlab()
        eng.eval('run initialize_matlab_variables.m', nargout=0)

    image_path = os.path.join(opts.data_dir, sub_dir_name)
    eng.save_classification_output_p(csv_detection_results_dir,
                                     opts.results_dir,
                                     sub_dir_name,
                                     image_path,
                                     opts.color_code_file, nargout=0)


def generate_output(network, opts, save_pre_process=True, network_output=True, post_process=True):
    cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    eng = matlab.engine.start_matlab()

    for cws_n in range(0, len(cws_sub_dir)):
        curr_cws_sub_dir = cws_sub_dir[cws_n]
        print(curr_cws_sub_dir, flush=True)
        sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
        csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)
        eng.eval('run initialize_matlab_variables.m', nargout=0)
        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

        if network_output:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
                assert ckpt, "No Checkpoint file found"
                # Restores from checkpoint
                saver.restore(sess, ckpt.model_checkpoint_path)
                print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
                generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                        logits_labels=logits_labels,
                                        csv_detection_results_dir=csv_detection_results_dir)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name,
                                csv_detection_results_dir=csv_detection_results_dir, eng=eng)

    return opts.results_dir


def generate_output_sub_dir(network, opts, sub_dir_name, save_pre_process=True, network_output=True, post_process=True):
    network.run_checks(opts=opts)
    logits, _ = network.inference(is_training=False)
    logits_labels = tf.argmax(logits[:, :, :, 0:network.num_of_classes], 3)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)
    csv_detection_results_dir = os.path.join(opts.detection_results_path, 'csv', sub_dir_name)

    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        assert ckpt, "No Checkpoint file found"
        # Restores from checkpoint
        saver.restore(sess, ckpt.model_checkpoint_path)
        print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)

        eng = matlab.engine.start_matlab()
        eng.eval('run initialize_matlab_variables.m', nargout=0)
        if save_pre_process:
            pre_process_images(opts=opts, sub_dir_name=sub_dir_name, eng=eng)

        if network_output:
            generate_network_output(opts=opts, sub_dir_name=sub_dir_name, network=network, sess=sess,
                                    logits_labels=logits_labels,
                                    csv_detection_results_dir=csv_detection_results_dir)

        if post_process:
            post_process_images(opts=opts, sub_dir_name=sub_dir_name,
                                csv_detection_results_dir=csv_detection_results_dir, eng=eng)

    return opts.results_dir