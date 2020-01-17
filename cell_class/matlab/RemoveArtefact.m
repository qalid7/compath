function doubleRGB = RemoveArtefact(I)
load('gm.mat');

od = -log((double(I)+1)/256);       % convert RGB to OD space
od = reshape(od,[],3);  

idx = cluster(gm,od);               % remove artefact
idx = reshape(idx,size(I,1),size(I,2));
im = reshape(I,[],3);
im = im2double(im);
flag = idx == 3;
im(flag,:) = 1;

doubleRGB = reshape(im,size(I,1),size(I,2),size(I,3));

end