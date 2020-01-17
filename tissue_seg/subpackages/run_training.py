import tensorflow as tf
import h5py
import os
import numpy as np
import scipy.io as sio
import time
from datetime import datetime

# from subpackages import NetworkOptions
# import mimo_net


def run_training(network, opts):
    # opts = NetworkOptions.NetworkOptions()
    # network = mimo_net.MIMONet(batch_size=opts.batch_size,
    #                            image_height=opts.image_height,
    #                            image_width=opts.image_width,
    #                            in_feat_dim=opts.in_feat_dim,
    #                            in_label_dim=opts.in_label_dim,
    #                            num_of_classes=opts.num_of_classes,
    #                            label_height=opts.label_height,
    #                            label_width=opts.label_width)
    train_data_file = os.path.join(opts.data_dir, opts.train_data_filename)
    valid_data_file = os.path.join(opts.data_dir, opts.valid_data_filename)
    print(train_data_file, flush=True)
    print(valid_data_file, flush=True)
    hft = h5py.File(train_data_file, 'r')
    data_set_t = hft.get('data')
    label_set_t = hft.get('labels')
    hfv = h5py.File(valid_data_file, 'r')
    data_set_v = hfv.get('data')
    label_set_v = hfv.get('labels')

    opts.num_examples_per_epoch_train, opts.image_height, opts.image_width, opts.in_feat_dim = data_set_t.shape
    _, opts.label_height, opts.label_width, opts.label_dim = label_set_t.shape
    opts.num_examples_per_epoch_valid, _, _, _ = data_set_v.shape

    train = np.arange(opts.num_examples_per_epoch_train)
    valid = np.arange(opts.num_examples_per_epoch_valid)

    network.run_checks(opts=opts)
    data_train = np.zeros([opts.batch_size, opts.image_height, opts.image_width, opts.in_feat_dim],
                          dtype=np.float32)
    label_train = np.zeros([opts.batch_size, opts.label_height, opts.label_width, opts.in_label_dim],
                           dtype=np.float32)

    train_count = int((len(train) / opts.batch_size) + 1)
    valid_count = int((len(valid) / opts.batch_size) + 1)

    global_step = tf.Variable(0.0, trainable=False)
    network.LearningRate = tf.placeholder(tf.float32)
    images, labels = network.random_crop(images=network.images, labels=network.labels)
    logits, _, _, _ = network.inference(images=images, is_training=True)

    imr0 = tf.concat(values=[logits[0:1, :, :, 0:1], logits[0:1, :, :, 1:2]], axis=1)
    imr1 = images[0:1, :, :, :]
    imr2 = tf.concat(values=[labels[0:1, :, :, 0:1], labels[0:1, :, :, 1:2]], axis=1)
    _ = tf.summary.image('Output_1', imr0)
    _ = tf.summary.image('Input_1', imr1)
    _ = tf.summary.image('label', imr2)

    loss = network.loss_function(global_step=global_step, labels=labels)
    correct_prediction = tf.equal(tf.argmax(labels[:, :, :, 0:2], 3),
                                  tf.argmax(logits[:, :, :, 0:2], 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    _ = tf.summary.scalar('Accuracy', accuracy)

    train_op = network.train()

    saver = tf.train.Saver(tf.global_variables(), max_to_keep=opts.num_of_epoch)

    avg_training_loss = 0.0
    avg_validation_loss = 0.0
    avg_training_accuracy = 0.0
    avg_validation_accuracy = 0.0

    config = tf.ConfigProto()
    # config = tf.ConfigProto(log_device_placement=True)
    # config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        summary_op = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'train'), sess.graph)
        valid_writer = tf.summary.FileWriter(os.path.join(opts.log_train_dir, 'valid'), sess.graph)
        init = tf.global_variables_initializer()
        ckpt = tf.train.get_checkpoint_state(opts.checkpoint_dir)
        curr_epoch = opts.current_epoch_num
        sess.run(init)
        if ckpt and ckpt.model_checkpoint_path:
            # Restores from checkpoint
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = int(ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]) + 1
            curr_epoch = int(global_step)
            print('Checkpoint file found at ' + ckpt.model_checkpoint_path, flush=True)
            workspace = sio.loadmat(os.path.join(opts.exp_dir, 'avg_training_loss_acc.mat'))
            avg_training_loss = np.array(workspace['avg_training_loss'])
            avg_validation_loss = np.array(workspace['avg_validation_loss'])
            avg_training_accuracy = np.array(workspace['avg_training_accuracy'])
            avg_validation_accuracy = np.array(workspace['avg_validation_accuracy'])
        else:
            print('No checkpoint file found', flush=True)

        for epoch in range(curr_epoch, opts.num_of_epoch):
            np.random.shuffle(train)
            np.random.shuffle(valid)
            # lr = 0.001/(int(epoch/10) + 1)
            lr = 0.001
            opts.current_epoch_num = global_step
            start = 0
            avg_loss = 0.0
            avg_accuracy = 0.0
            start_time = time.time()
            for step in range(train_count):
                # for step in range(1):
                end = start + opts.batch_size
                start_time_step = time.time()
                temp_indices = train[start:end]
                np.random.shuffle(temp_indices)
                indices = np.sort(temp_indices)
                data_set_t.read_direct(data_train, np.s_[indices, :, :, :])
                label_set_t.read_direct(label_train, np.s_[indices, :, :, :])
                data_train_float32 = np.divide(np.float32(data_train), 65535.0)
                label_train_float32 = np.float32(label_train)
                if (step % int(25) == 0) or (step == 0):
                    _, _, logits_out, summary_str, _, loss_value, accuracy_value = sess.run(
                        [images, labels, logits, summary_op, train_op, loss, accuracy],
                        feed_dict={network.images: data_train_float32,
                                   network.labels: label_train_float32,
                                   network.LearningRate: lr})
                    train_writer.add_summary(summary_str, step + epoch * train_count)
                    inter = {'logits': logits_out,
                             'input': data_train_float32,
                             'label': label_train_float32}
                    sio.savemat(os.path.join(opts.exp_dir, 'inter_train.mat'), inter)
                else:
                    _, _, _, _, loss_value, accuracy_value = sess.run(
                        [images, labels, logits, train_op, loss, accuracy],
                        feed_dict={network.images: data_train_float32,
                                   network.labels: label_train_float32,
                                   network.LearningRate: lr})

                if end + opts.batch_size > len(train) - 1:
                    end = len(train) - opts.batch_size

                start = end
                avg_loss += loss_value
                avg_accuracy += accuracy_value
                duration = time.time() - start_time_step
                format_str = (
                    '%s: epoch %d, step %d/ %d, Training Loss = %.2f, Training Accuracy = %.2f, (%.2f sec/step)')
                print(
                    format_str % (
                        datetime.now(), epoch + 1, step + 1, train_count,
                        loss_value, accuracy_value, float(duration)), flush=True)

            training_loss = avg_loss / train_count
            training_accuracy = avg_accuracy / train_count

            start = 0
            avg_loss = 0.0
            avg_accuracy = 0.0
            for step in range(valid_count):
                # for step in range(1):
                end = start + opts.batch_size
                start_time_step = time.time()
                indices = np.sort(valid[start:end])
                data_set_v.read_direct(data_train, np.s_[indices, :, :, :])
                label_set_v.read_direct(label_train, np.s_[indices, :])
                data_train_float32 = np.divide(np.float32(data_train), 65535.0)
                label_train_float32 = np.float32(label_train)
                if (step % int(1) == 0) or (step == 0):
                    _, _, logits_out, summary_str, _, loss_value, accuracy_value = sess.run(
                        [images, labels, logits, summary_op, train_op, loss, accuracy],
                        feed_dict={network.images: data_train_float32,
                                   network.labels: label_train_float32,
                                   network.LearningRate: lr})
                    valid_writer.add_summary(summary_str, step + epoch * train_count)
                    inter = {'logits': logits_out,
                             'input': data_train_float32,
                             'label': label_train_float32}
                    sio.savemat(os.path.join(opts.exp_dir, 'inter_valid.mat'), inter)
                else:
                    _, _, _, _, loss_value, accuracy_value = sess.run(
                        [images, labels, logits, train_op, loss, accuracy],
                        feed_dict={network.images: data_train_float32,
                                   network.labels: label_train_float32,
                                   network.LearningRate: lr})

                if end + opts.batch_size > len(valid) - 1:
                    end = len(valid) - opts.batch_size

                start = end
                avg_loss += loss_value
                avg_accuracy += accuracy_value
                duration = time.time() - start_time_step
                format_str = (
                    '%s: epoch %d, step %d/ %d, Validation Loss = %.2f, '
                    'Validation Accuracy = %.2f, (%.2f sec/step)')
                print(
                    format_str % (
                        datetime.now(), epoch + 1, step + 1, valid_count, loss_value, accuracy_value,
                        float(duration)), flush=True)
            validation_loss = avg_loss / valid_count
            validation_accuracy = avg_accuracy / valid_count

            checkpoint_path = os.path.join(opts.checkpoint_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=global_step)
            global_step = global_step + 1
            # Average loss on training and validation datasets.
            duration = time.time() - start_time
            format_str = (
                '%s: epoch %d, Training Loss = %.2f, Validation Loss = %.2f, '
                'Training Accuracy = %.2f, Validation Accuracy = %.2f, (%.2f sec/epoch)')
            print(format_str % (
                datetime.now(), epoch + 1, training_loss, validation_loss,
                training_accuracy, validation_accuracy, float(duration)), flush=True)
            if epoch == 0:
                avg_training_loss = [int(training_loss)]
                avg_validation_loss = [int(validation_loss)]
                avg_training_accuracy = [int(training_accuracy * 100)]
                avg_validation_accuracy = [int(validation_accuracy * 100)]
            else:
                avg_training_loss = np.append(avg_training_loss, [int(training_loss)])
                avg_validation_loss = np.append(avg_validation_loss, [int(validation_loss)])
                avg_training_accuracy = np.append(avg_training_accuracy, [int(training_accuracy * 100)])
                avg_validation_accuracy = np.append(avg_validation_accuracy, [int(validation_accuracy * 100)])
            avg_training_loss_acc_dict = {'avg_training_loss': avg_training_loss,
                                          'avg_validation_loss': avg_validation_loss,
                                          'avg_training_accuracy': avg_training_accuracy,
                                          'avg_validation_accuracy': avg_validation_accuracy,
                                          }
            sio.savemat(file_name=os.path.join(opts.exp_dir, 'avg_training_loss_acc.mat'),
                        mdict=avg_training_loss_acc_dict)
            print(avg_training_loss, flush=True)
            print(avg_validation_loss, flush=True)
            print(avg_training_accuracy, flush=True)
            print(avg_validation_accuracy, flush=True)
        return network
