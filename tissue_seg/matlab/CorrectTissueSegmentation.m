function [ corrected ] = CorrectTissueSegmentation( results_dir, sub_dir_name, image_path )
%CORRECTTISSUESEGMENTATION Summary of this function goes here
%   Detailed explanation goes here

mat = load(fullfile(results_dir, 'mat', sub_dir_name, 'Ss1.mat'));
if isfield(mat, 'mat')
    mat = mat.mat;
end
Ss1_im = imread(fullfile(image_path, 'Ss1.jpg'));
figure, imshow(im2double(Ss1_im).*repmat(mat.BinLabel, [1,1,3]));
f = figure;
imshow(Ss1_im);

choice = questdlg('Would you like to correct segmentation?', ...
	'Correct Segmentation', ...
	'Yes','No','No');

switch choice
    case 'Yes'
        corrected = true;
        GT.imfreehandles = [];
        GT.im = Ss1_im;
        GT.Mask = zeros(size(Ss1_im,1), size(Ss1_im,2));
        cla;
        ax = axes;
        GT.h_im = imagesc(ax,Ss1_im);
        axis image;
        set(f,'UserData', GT);
        h = 1;
        while(~isempty(h))
            h = imfreehand(gca);
            GT.imfreehandles{end+1} = h;
            set(f,'UserData', GT);
        end
        GT = get(f, 'UserData');
        v = [];
        if ~isempty(GT.imfreehandles)
            imfreehandles = GT.imfreehandles;
            for i = 1:length(imfreehandles)
                if isvalid(imfreehandles{i})
                    v = [v;i];  %#ok<AGROW>
                end
            end
            GT.imfreehandles = imfreehandles(v);
            BW = zeros(size(GT.im,1), size(GT.im,2));
            for i = 1:length(GT.imfreehandles)
                BW = createMask(GT.imfreehandles{i}, GT.h_im)|BW;
                GT.Position{i} = getPosition(GT.imfreehandles{i});
            end
            GT.Mask = BW;
        end
        mat.BinLabel = GT.Mask;
        save(fullfile(results_dir, 'mat', sub_dir_name, 'Ss1.mat'), 'mat');
        annotated_image = im2double(Ss1_im).*repmat(mat.BinLabel, [1 1 3]);
        imwrite(annotated_image, fullfile(results_dir, 'annotated_images', [sub_dir_name, '.png']), 'png');
        close all;
    otherwise
        corrected = false;
        close all;
        return;
end

end

