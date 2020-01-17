function Save_Detection_Output(results_dir, sub_dir_name, h5_file_name, image_path_full)
    if ~exist(...
            fullfile(results_dir, 'annotated_images', sub_dir_name, ...
            [h5_file_name(1:end-2), 'png']), 'file')
        strength = 3;
   
        output = h5load(fullfile(results_dir, 'h5', sub_dir_name, h5_file_name),'output');
        
        detection = FindLocalMaximaMaxClique(output,12,0.15);
        V = cell(size(detection,1),3);
        detection_table = cell2table(V);
        if ~isempty(detection)
            detection_table.V1 = repmat({'None'},[size(detection,1),1]);
            detection_table.V2 = detection(:,1);
            detection_table.V3 = detection(:,2);
            writetable(detection_table, fullfile(results_dir, 'csv', sub_dir_name, [h5_file_name(1:end-2), 'csv']));
        else
            fileID = fopen(fullfile(results_dir, 'csv', sub_dir_name, [h5_file_name(1:end-2), 'csv']), 'w');
            fprintf(fileID, 'V1,V2,V3');
        end
        image = imread(image_path_full);
        image = annotate_image_with_class(image, detection, [0 1 0], strength);
        imwrite(image, fullfile(results_dir, 'annotated_images', sub_dir_name, [h5_file_name(1:end-2), 'png']), 'png');
        h5save(fullfile(results_dir, 'h5', sub_dir_name, h5_file_name),  detection, 'detection');
%         parsave_csv(fullfile(results_dir, 'csv', sub_dir_name, [h5_file_name(1:end-2), 'csv']),  detection);
        close all;
    else
        fprintf('Already Processed %s\n', ...
            fullfile(results_dir, 'annotated_images', sub_dir_name, ...
            [h5_file_name(1:end-2), 'png']))
    end
end