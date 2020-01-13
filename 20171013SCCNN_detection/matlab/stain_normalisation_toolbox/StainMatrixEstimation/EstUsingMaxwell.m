function [ M ] = EstUsingMaxwell( I, clusters, verbose )
%MAXWELLSTAINSEPARATION Summary of this function goes here
%   Detailed explanation goes here

    if nargin < 2
        clusters = 3;
        verbose = 0;
    end
    
    if nargin < 3
        verbose = 0;
    end

    I = im2double(I);
    columnImage = reshape(I, [], 3);
    
    columnImage = columnImage(~any(columnImage==0, 2), :);
    columnImage = columnImage(~all(columnImage<0.05, 2), :);
    columnImage = columnImage(~all(columnImage>0.8, 2), :);
    
    columnImage_OD = -log(columnImage);
    
    MaxMat = [(2.^-0.5) -(2.^-0.5) 0; -(6.^-0.5) -(6.^-0.5) ((2/3).^0.5)];
    MaxwellImage = (MaxMat*((0-columnImage_OD)./repmat(sum(abs(0-columnImage_OD), 2), [1 3]))')';
    
    sampleIndices = round(linspace(1, size(columnImage, 1), 20000));
    
    %labels = emgm([columnImage_OD(sampleIndices, :) MaxwellImage(sampleIndices, :)]', clusters)';
    [labels, model] = emgm(MaxwellImage(sampleIndices, :)', clusters);
    labels = labels';
    
    M = [sqrt(0.5) -sqrt(1/6) 1/3; -sqrt(0.5) -sqrt(1/6) 1/3; 0 2*sqrt(1/6) 1/3]*[model.mu; ones(1, clusters)];
    
    M = M-(2/3);
    
    for i=1:clusters
        M(:, i) = M(:, i)*mean(sum(abs(0-columnImage_OD(sampleIndices(labels==i), :)), 2));
    end
    
    M = pinv(M'*M)*M';
    
    if verbose
        figure; imagesc(I);
    
        figure; subplot(1, 2, 1); scatter(MaxwellImage(sampleIndices, 1), MaxwellImage(sampleIndices, 2), 5, columnImage(sampleIndices, :));
        subplot(1, 2, 2); scatter(MaxwellImage(sampleIndices, 1), MaxwellImage(sampleIndices, 2), 5, labels);
    
        figure; hist3(MaxwellImage, [100 100]);
    
        histplot = get(gca, 'Children');
        XRange = get(histplot, 'XData');
        YRange = get(histplot, 'YData');
    
        CData = (exp(-MaxMat\[XRange(:) YRange(:)]'))';
        CData = CData-repmat(min(CData, [], 2), [1 3]);
        CData = CData./repmat(max(CData, [], 2), [1 3]);
        CData = reshape(CData, [size(XRange) 3]);
    
        set(histplot, 'CData', CData);
    end
    
    
end

