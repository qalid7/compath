import os
import glob

dir_path = 'R:/tracerx/tracerx100/IHC_diagnostic/results/detection/20171013/csv'
dir_pattern = '*.ndpi'
file_pattern = '*.csv'
file_ext = '.csv'
tissue_mat_dir_path = 'D:/tmp/cws-ihc/results/tissue_segmentation/20171019/mat'

cws_sub_dir = sorted(glob.glob(os.path.join(dir_path, dir_pattern)))

for cws_n in range(0, len(cws_sub_dir)):
    curr_cws_sub_dir = cws_sub_dir[cws_n]
    print(curr_cws_sub_dir)
    sub_dir_name = os.path.basename(os.path.normpath(curr_cws_sub_dir))
    files = sorted(glob.glob(os.path.join(dir_path, sub_dir_name, file_pattern)))
    for i in range(len(files)):
        file_base_name = os.path.basename(files[i])
        filename = file_base_name.split(file_ext)[0]
        if not os.path.isfile(os.path.join(tissue_mat_dir_path, sub_dir_name, filename + '.mat')):
            # print(files[i])
            os.remove(files[i])

