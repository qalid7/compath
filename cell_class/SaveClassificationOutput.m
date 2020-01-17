clc
clear
addpath('C:\Users\adminsraza\Documents\MATLAB\export_fig');
addpath matlab\
%%
filepath = 'D:\Shan\MyCodes\TracerX\CellDetection\SCCNN\SCCNN_v2\ExpDir\TCGA-05-4389-01A-01-BS1';
classifier_output = 'D:\Shan\MyCodes\TracerX\CellDetection\SCCNN\SCCNN_v2\ExpDir\TCGA-05-4389-01A-01-BS1';
imagefolder = 'Z:\TCGA\Lung\data\raw\LUAD_20x\TCGA-05-4389-01A-01-BS1';
files = dir(fullfile(filepath, '*.csv'));
%%
warning('off');
if ~exist(fullfile(classifier_output, 'classification'), 'dir')
    mkdir(fullfile(classifier_output, 'classification'));
end
parfor i = 1:length(files)
    fprintf('%s\n', files(i).name);
    A = importdata(fullfile(files(i).folder, files(i).name));
    image = imread(fullfile(imagefolder, [files(i).name(1:end-3), 'jpg']));
    h = figure('Visible', 'off');
    warning('off', 'Images:initSize:adjustingMag');
    imshow(image,[]);
    if isfield(A, 'data')
        detection = A.data;
        mat = load(fullfile(classifier_output, [files(i).name(1:end-4), ...
            '_classification.mat']));
        cell_ids = mat.cell_ids;
        output = mat.output;
        C = unique(cell_ids);
        class = zeros(length(C),1);
        for j = 1:length(C)
            class(j) = mode(mat.output(mat.cell_ids==C(j)));
        end
        hold on;
        plot(detection(class==1,1),detection(class==1,2),'.y','markersize',5);
        plot(detection(class==2,1),detection(class==2,2),'.b','markersize',5);
        plot(detection(class==3,1),detection(class==3,2),'.g','markersize',5);
        plot(detection(class==4,1),detection(class==4,2),'.r','markersize',5);
        hold off;
    end
    imagedata = export_fig(gca, '-m3');
    imagedata = imresize(imagedata, [size(image,1),size(image,2)]);
    imwrite(imagedata, fullfile(classifier_output, 'annotated_images', [mat_file_name(1:end-3), 'png']), 'png');
    close(gcf);
end