files = dir(fullfile(matlab_input.input_path, 'Ss1.jpg'));
if ~exist(fullfile(matlab_input.output_path, 'pre_processed', matlab_input.sub_dir_name), 'dir')
    mkdir(fullfile(matlab_input.output_path, 'pre_processed', matlab_input.sub_dir_name));
end
fprintf('Pre-Processing %s\n', matlab_input.sub_dir_name);
for i = 1:length(files)
    TargetImage = imread('TargetImage.png');
    if ~exist(fullfile(matlab_input.output_path, 'pre_processed', matlab_input.sub_dir_name, ...
            [files(i).name(1:end-3), 'mat']), 'file')        
        I = imread(fullfile(matlab_input.input_path, files(i).name));
%         I = Retinex(I);
        I = im2uint8(NormReinhard( I, TargetImage));
        features = matlab_input.feat;
        
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
        parsave(fullfile(matlab_input.output_path, 'pre_processed', matlab_input.sub_dir_name, ...
            [files(i).name(1:end-3), 'mat']), feat);
    end
end