function Save_Detection_Output_p(results_dir, sub_dir_name, image_path)
    if ~exist(fullfile(results_dir, 'annotated_images'), 'dir')
        mkdir(fullfile(results_dir, 'annotated_images'));
    end
    addpath('C:\Users\adminsraza\Documents\MATLAB\export_fig');
    files = dir(fullfile(results_dir, 'mat', sub_dir_name, '*.mat'));
    warning('off');
    for i = 1:length(files)
        fprintf('%s\n', files(i).name);
        mat_file_name = files(i).name;
        image_path_full = fullfile(image_path, [files(i).name(1:end-3), 'jpg']);
        Save_Detection_Output(results_dir, sub_dir_name, mat_file_name, image_path_full);
    end
end