function Save_Classification_Output_p(csv_detection_results_dir, results_dir, sub_dir_name, image_path, color_code_file)
%     addpath('C:\Users\adminsraza\Documents\MATLAB\export_fig');
    files = dir(fullfile(results_dir, 'mat', sub_dir_name, '*.mat'));
    warning('off');
    parfor i = 1:length(files)
        mat_file_name = [files(i).name(1:end-3), 'mat'];
        csv_file_name = fullfile(csv_detection_results_dir, [files(i).name(1:end-3), 'csv']);
        image_path_full = fullfile(image_path, [files(i).name(1:end-3), 'jpg']);
%         fprintf('%s\n', image_path_full);
        Save_Classification_Output(results_dir, sub_dir_name, mat_file_name, image_path_full, csv_file_name, color_code_file);
    end
end