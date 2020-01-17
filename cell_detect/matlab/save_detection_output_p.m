function save_detection_output_p(results_dir, sub_dir_name, image_path)
    files = dir(fullfile(results_dir, 'h5', sub_dir_name, '*.h5'));
    parfor i = 1:length(files)
        fprintf('%s\n', files(i).name);
        h5_file_name = files(i).name;
        image_path_full = fullfile(image_path, [files(i).name(1:end-2), 'jpg']);
        Save_Detection_Output(results_dir, sub_dir_name, h5_file_name, image_path_full);
    end
end