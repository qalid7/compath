clc
clear
close all;
addpath('~/matlab');
%%
%%modifeid on 1902177 to discard white rubbish areas outside (or at the
%%boundary) of tissue seg.

%1. change all these paths:
classification_dir = ''; %output of the pipeline, cell classification dir
cws_dir = ''; %original tiles data
save_clean = ''; %output of this code, cleaned cell classification
save_dir = fullfile(save_clean, 'summary', 'cellnumbers-200124');
save_filename = ''; %name of the summary table

%2. select colorcodes
colorcodes = readtable("~/matlab/colorcodes/HE_Fib_Lym_Tum_Others.csv");

%3. what are your files? 
files = dir(fullfile(classification_dir, 'mat', '*.ndpi'));

%4. make a table, you can skip or change this according to what's needed:
M_f = cell2table({'',0,0,0,0,0,0,0,0,0,0,0}, 'VariableNames', {'FileName', ...
    'fibroblasts', 'lymphocytes', 'tumour', 'othercells', ...
    'fibroblasts_per', 'lymphocytes_per', 'tumour_per', 'othercells_per',...
    'fibroblasts_per_excl_artifacts', 'lymphocytes_per_excl_artifacts', 'tumour_per_excl_artifacts'});


strength = 5;
%%

%making output dirs:
if ~exist(save_dir, 'dir')
    mkdir(save_dir);
end
if ~exist(save_clean, 'dir')
    mkdir(save_clean);
    mkdir(fullfile(save_clean, 'csv'));
    mkdir(fullfile(save_clean, 'mat'));
    mkdir(fullfile(save_clean, 'annotated_images'));
end

%main loop
for i = 1:length(files)
    fprintf('Processing %d of %d\n', i, length(files));
    M_f.FileName(i,1)= {files(i).name};
    mat_dir = fullfile(classification_dir, 'mat', files(i).name);
    csv_dir = fullfile(classification_dir, 'csv', files(i).name);
    DaImg_dir = fullfile(cws_dir, files(i).name);
    
    mkdir(fullfile(save_clean, 'csv', files(i).name));
    mkdir(fullfile(save_clean, 'mat', files(i).name));
    mkdir(fullfile(save_clean, 'annotated_images', files(i).name));
    
    mat_files = dir(fullfile(mat_dir, '*.mat'));
    DaImg_files = dir(fullfile(DaImg_dir, '*.jpg'));
    total_fibroblasts = 0;
    total_lymphocytes = 0;
    total_tumourcells = 0;
    total_othercells = 0;

%change to 'for' to debug/avoid parfor
    parfor j = 1:length(mat_files)
        mat = load(fullfile(mat_dir, mat_files(j).name));
        csv_table = readtable(fullfile(csv_dir, [mat_files(j).name(1:end-3), 'csv']));
        if isfield(mat, 'mat')
            mat = mat.mat;
        end
		
		if ~isfield(mat, 'class')
            fprintf('%s/%s - NO mat.class! \n', files(i).name, mat_files(j).name);
        end
        
        if isfield(mat, 'class')
		
%         class = mat.class;
        DaImg = imread(fullfile(DaImg_dir, [mat_files(j).name(1:end-3), 'jpg']));
        DaImgr = min(DaImg, [], 3);

	%this is just an example, modify accordingly 
        DaImgr = bwareaopen(DaImgr>200, 10); 
%         S = regionprops(CC, 'Area');
        
        %discard cells in the white, outside the area
        linearInd = sub2ind(size(DaImgr), csv_table.V3, csv_table.V2);
        to_discard = DaImgr(linearInd);
        csv_table(to_discard, :) = [];
		
		%why isn't passing mat.class
        %very rare but:
        %if height(csv_table)>5
        %but the problem is csv_table dont have a mat.class
		
        mat.class(to_discard, :) = [];
        
        detection = [csv_table.V2, csv_table.V3];
        class = mat.class;
        image = DaImg;
        for c = 1:height(colorcodes)
            image = annotate_image_with_class(image, detection(class==c,:), ...
                hex2rgb(colorcodes.color{c}), strength);
        end
        writetable(csv_table, fullfile(save_clean, 'csv', files(i).name, [mat_files(j).name(1:end-3), 'csv']));
        parsave(fullfile(save_clean, 'mat', files(i).name, mat_files(j).name), mat);
        imwrite(image, fullfile(save_clean, 'annotated_images', files(i).name, [mat_files(j).name(1:end-3), 'jpg']));
        fibroblasts = sum(class==1);
        lymphocytes = sum(class==2);
        tumour = sum(class==3);
        othercells = sum(class==4);
        
        total_fibroblasts = total_fibroblasts + fibroblasts;
        total_lymphocytes = total_lymphocytes + lymphocytes;
        total_tumourcells = total_tumourcells + tumour;
        total_othercells = total_othercells + othercells;
		end
    end
    
    total = total_fibroblasts + total_lymphocytes + ...
        total_tumourcells + total_othercells;
   
    
    M_f.fibroblasts(i,1)= total_fibroblasts;
    M_f.lymphocytes(i,1)= total_lymphocytes;
    M_f.tumour(i,1)= total_tumourcells;
    M_f.othercells(i,1)= total_othercells;
    
    
    M_f.fibroblasts_per(i,1)= total_fibroblasts/total*100;
    M_f.lymphocytes_per(i,1)= total_lymphocytes/total*100;
    M_f.tumour_per(i,1)= total_tumourcells/total*100;
    M_f.othercells_per(i,1)= total_othercells/total*100;
    
    total = total_fibroblasts + total_lymphocytes + ...
        total_tumourcells;
    
    M_f.fibroblasts_per_excl_artifacts(i,1)= total_fibroblasts/total*100;
    M_f.lymphocytes_per_excl_artifacts(i,1)= total_lymphocytes/total*100;
    M_f.tumour_per_excl_artifacts(i,1)= total_tumourcells/total*100;
    
    fprintf('%s - DONE!\n', files(i).name);
end

writetable(M_f, fullfile(save_dir, save_filename));
