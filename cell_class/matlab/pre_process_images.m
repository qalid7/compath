function pre_process_images(matlab_input)

output_path = matlab_input.output_path;
sub_dir_name = matlab_input.sub_dir_name;
tissue_segment_dir = matlab_input.tissue_segment_dir;
input_path = matlab_input.input_path;
features = matlab_input.feat;
if ~exist(fullfile(output_path, 'pre_processed', sub_dir_name), 'dir')
    mkdir(fullfile(output_path, 'pre_processed', sub_dir_name));
end
if ~isempty(tissue_segment_dir)
    files_tissue = dir(fullfile(tissue_segment_dir, 'mat', sub_dir_name, 'Da*.mat'));
else
    files_tissue = dir(fullfile(input_path, 'Da*.jpg'));
end
display(matlab_input)
display(files_tissue)
parfor i = 1:length(files_tissue)
    if ~exist(fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']), 'file')
        fprintf('%s\n', fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']));
%         TargetImage = imread('Target.png');
        I = imread(fullfile(input_path, [files_tissue(i).name(1:end-3), 'jpg']));
        I = Retinex(I);         % adjust using Retinex
%         I = im2uint8(NormReinhard( I, TargetImage));
        
        feat = [];
        for iFeatures = 1:length(features)
            switch features{iFeatures}
                case 'rgb'
                    feat = cat(3,feat,single(I));
                case 'lab'
                    feat = cat(3,feat,single(rgb2lab(I)));
                case 'h'
                    doubleRGB = RemoveArtefact(I);
                    
                    colourMat = EstUsingSCD(doubleRGB);
                    [ DCh ] = Deconvolve( I, colourMat );
                    [ H ] = PseudoColourStains( DCh, colourMat );
                    H = rgb2gray(H);
                    feat = cat(3,feat,single(H));
                case 'e'
                    doubleRGB = RemoveArtefact(I);
                    colourMat = EstUsingSCD(doubleRGB);
                    [ DCh ] = Deconvolve( I, colourMat );
                    [ ~, E ] = PseudoColourStains( DCh, colourMat );
                    E = rgb2gray(E);
                    feat = cat(3,feat,single(E));
                case 'he'
                    doubleRGB = RemoveArtefact(I);
                    colourMat = EstUsingSCD(doubleRGB);
                    [ DCh ] = Deconvolve( I, colourMat );
                    [ H, E ] = PseudoColourStains( DCh, colourMat );
                    H = rgb2gray(H);
                    E = rgb2gray(E);
                    feat = cat(3,feat,single(H),single(E));
                case 'br'
                    BR = BlueRatioImage(I);
                    feat = cat(3,feat,single(BR));
                case 'grey'
                    grey = rgb2gray(I);
                    feat = cat(3,feat,single(grey));
            end
        end
        h5save(fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']), feat, 'feat');
    
    else
        fprintf('Already Pre-Processed %s\n', ...
            fullfile(output_path, 'pre_processed', sub_dir_name, ...
            [files_tissue(i).name(1:end-3), 'h5']))
    end
end
end