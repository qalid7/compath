function Save_Classification_Output(results_dir, sub_dir_name, mat_file_name, image_path_full, csv_file_name, color_code_file)
if ~exist(fullfile(results_dir, 'annotated_images', sub_dir_name, [mat_file_name(1:end-3), 'png']), 'file')
    strength = 5;
    A = readtable(csv_file_name);
    image = imread(image_path_full);
    class = [];
    colorcodes = readtable(fullfile(fileparts(mfilename('fullpath')), 'colorcodes', color_code_file));
    if ~isempty(A)
        detection = [A.V2, A.V3];
        mat = load(fullfile(results_dir, 'mat', sub_dir_name, mat_file_name));
        if isfield(mat, 'mat')
            mat = mat.mat;
        end
        cell_ids = mat.cell_ids;
        C = unique(cell_ids);
        class = zeros(length(C),1);
        for j = 1:length(C)
            class(j) = mode(mat.output(mat.cell_ids==C(j)));
        end
        for c = 1:height(colorcodes)
            image = annotate_image_with_class(image, detection(class==c,:), ...
                hex2rgb(colorcodes.color{c}), strength);
        end
        A.V1 = colorcodes.class(class);
        writetable(A, fullfile(results_dir, 'csv', sub_dir_name, [mat_file_name(1:end-3), 'csv']));
    else
        fileID = fopen(fullfile(results_dir, 'csv', sub_dir_name, [mat_file_name(1:end-3), 'csv']), 'w');
        fprintf(fileID, 'V1,V2,V3');
    end
    mat.class = class;
    imwrite(image, fullfile(results_dir, 'annotated_images', sub_dir_name, [mat_file_name(1:end-3), 'png']), 'png');
    parsave_mat(fullfile(results_dir, 'mat', sub_dir_name, mat_file_name),  mat);
    fprintf('saved %s\n', fullfile(results_dir, 'annotated_images', sub_dir_name, [mat_file_name(1:end-3), 'png']));
else
    fprintf('Already saved %s\n', fullfile(results_dir, 'annotated_images', sub_dir_name, [mat_file_name(1:end-3), 'png']));
end