import scipy.io as sio
import numpy as np
import h5py
import os
import glob

save_path = os.path.normpath("D:/Shan/MyCodes/TracerX/TissueSegmentation/Code/PrepareDataSet/TissueSeg_IHC")
main_file_path = os.path.normpath("D:/Shan/MyCodes/TracerX/TissueSegmentation/Code/PrepareDataSet/TissueSeg_IHC/mat")
Train_files = sorted(glob.glob(os.path.join(main_file_path, 'Train*')))
Valid_files = sorted(glob.glob(os.path.join(main_file_path, 'Valid*')))
Train_Data_name = 'TrainData-IHC-171113.h5'
Valid_Data_name = 'ValidData-IHC-171113.h5'


def read_mat_file(file_name):
    workspace = sio.loadmat(file_name)
    images = workspace['images']
    im_data = images['data'][0][0]
    im_labels = images['labels'][0][0]
    if len(im_data.shape) == 3:
        im_data = im_data[:, :, :, np.newaxis]
        im_labels = im_labels[:, :, :, np.newaxis]
    im_data = np.transpose(im_data, [3, 0, 1, 2])
    im_labels = np.transpose(im_labels, [3, 0, 1, 2])
    labels0 = im_labels[:, :, :, 0] - 1
    labels1 = 2 - im_labels[:, :, :, 0]
    labels2 = im_labels[:, :, :, 1]
    im_labels = np.concatenate((labels1[:, :, :, np.newaxis], labels0[:, :, :, np.newaxis],
                                labels2[:, :, :, np.newaxis], labels2[:, :, :, np.newaxis]), axis=3)
    return im_data, im_labels


data, labels = read_mat_file(file_name=Train_files[0])

hf = h5py.File(os.path.join(save_path, Train_Data_name), 'w-')
data_set = hf.create_dataset("data", data.shape,
                             maxshape=(None, data.shape[1], data.shape[2], data.shape[3]),
                             dtype='float32')
label_set = hf.create_dataset("labels", labels.shape,
                              maxshape=(None, labels.shape[1], labels.shape[2], labels.shape[3]),
                              dtype='float32')

for Train_n in range(0, len(Train_files)):
    curr_Train_file = Train_files[Train_n]
    print('Processing ' + curr_Train_file)
    data, labels = read_mat_file(file_name=Train_files[Train_n])
    ExistingPatches, _, _, _ = data_set.shape
    if Train_n == 0:
        ExistingPatches = 0
    data_set.resize((int(ExistingPatches + data.shape[0]), data.shape[1], data.shape[2], data.shape[3]))
    label_set.resize((int(ExistingPatches + data.shape[0]), labels.shape[1], labels.shape[2], labels.shape[3]))
    data_set[int(ExistingPatches):int(ExistingPatches + data.shape[0]), :, :, :] = data[:, :, :, :]
    label_set[int(ExistingPatches):int(ExistingPatches + data.shape[0]), :, :, :] = labels[:, :, :, :]
    del data, labels

hf.close()

data, labels = read_mat_file(file_name=Valid_files[0])
hf = h5py.File(os.path.join(save_path, Valid_Data_name), 'w-')
data_set = hf.create_dataset("data", data.shape,
                             maxshape=(None, data.shape[1], data.shape[2], data.shape[3]),
                             dtype='float32')
label_set = hf.create_dataset("labels", labels.shape,
                              maxshape=(None, labels.shape[1], labels.shape[2], labels.shape[3]),
                              dtype='float32')

for Valid_n in range(0, len(Valid_files)):
    curr_Valid_file = Valid_files[Valid_n]
    print('Processing ' + curr_Valid_file)
    data, labels = read_mat_file(file_name=Valid_files[Valid_n])
    ExistingPatches, _, _, _ = data_set.shape
    if Valid_n == 0:
        ExistingPatches = 0
    data_set.resize((int(ExistingPatches + data.shape[0]), data.shape[1], data.shape[2], data.shape[3]))
    label_set.resize((int(ExistingPatches + data.shape[0]), labels.shape[1], labels.shape[2], labels.shape[3]))
    data_set[int(ExistingPatches):int(ExistingPatches + data.shape[0]), :, :, :] = data[:, :, :, :]
    label_set[int(ExistingPatches):int(ExistingPatches + data.shape[0]), :, :, :] = labels[:, :, :, :]
    del data, labels

hf.close()
