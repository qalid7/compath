import os

import matlab.engine
import glob
import scipy.io as sio
import math
import pandas as pd
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
                                     data_dir='D:\\tmp\\cws-ihc\\data\\cws',
                                     results_dir='D:\\tmp\\cws-ihc\\results\\'
                                                 'tissue_segmentation',
                                     current_epoch_num=0,
                                     file_name_pattern='*.ndpi',
                                     pre_process=True,
                                     )

opts.results_dir = (os.path.join(opts.results_dir, '20171019'))

eng = matlab.engine.start_matlab()
eng.eval('run initialize_matlab_variables.m', nargout=0)
cws_sub_dir = sorted(glob.glob(os.path.join(opts.data_dir, opts.file_name_pattern)))

for cws_n in range(0, len(cws_sub_dir)):
    curr_cws_sub_dir = cws_sub_dir[cws_n]
    print(curr_cws_sub_dir)
    sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
    image_path = os.path.join(opts.data_dir, sub_dir_name)
    corrected = eng.CorrectTissueSegmentation(opts.results_dir, sub_dir_name, image_path)
    if corrected is True:
        files = sorted(glob.glob(os.path.join(opts.data_dir, sub_dir_name, 'Ss1.jpg')))
        i = 0
        for f in glob.glob(os.path.join(opts.results_dir, 'mat', sub_dir_name, 'Da*.mat')):
            os.remove(f)
        os.remove(os.path.join(opts.results_dir, 'csv', sub_dir_name + '.csv'))
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

