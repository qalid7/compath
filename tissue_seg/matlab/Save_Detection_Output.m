function Save_Detection_Output(results_dir, sub_dir_name, mat_file_name, image_path_full)
    mat = load(fullfile(results_dir, 'mat', sub_dir_name, mat_file_name));
    [~,BinLabel] = max(mat.output, [],3);
    BinLabel = BinLabel>1;   
    BinLabel = imdilate(BinLabel, strel('disk', 5));
    BinLabel = imfill(BinLabel, 'holes');
    mat.BinLabel = bwareaopen(BinLabel, 750);
    im = imread(image_path_full);
    annotated_image = im2double(im).*repmat(mat.BinLabel, [1 1 3]);
    imwrite(annotated_image, fullfile(results_dir, 'annotated_images', [sub_dir_name, '.png']), 'png');
    parsave_mat(fullfile(results_dir, 'mat', sub_dir_name, mat_file_name),  mat);
    close all;
end