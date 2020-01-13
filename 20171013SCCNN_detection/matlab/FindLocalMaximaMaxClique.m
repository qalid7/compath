function [p,grayImage,thres] = FindLocalMaximaMaxClique(grayImage,graphGroupingDistance,bgThres)
% FindLocalMaximaMaxClique works like its name suggests.
% Inputs: 
%   grayImage: an 2D image
%   graphGroupingDistance: neighbour local maxima are merged together if they
%                  are less than graphDistance apart
%
% Outputs
%   p: coordinates of local maxima 
%
% Korsuk Sirinukunwattana
% BIAlab, Department of Computer Science, University of Warwick
% 2015 - 4 - 19 add rowwise local maxima
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% preprocess by removing salt&pepper noise and smoothing
filt = fspecial('gaussian', [3,3],1);
grayImage = conv2(grayImage,filt,'same') ;

% threshold
bgThres = bgThres*max(grayImage(:));
thresInit = max([min(max(grayImage,[],1))  min(max(grayImage,[],2))]);
if thresInit <= 0.18
    thresInit = 0.18;
end

if nargin < 3
    thres = thresInit;
else
    thres = bgThres;
end


% columnwise local maxima
[~,locs]=findpeaks(grayImage(:),'MinPeakHeight', thres, 'MinPeakDistance', graphGroupingDistance);
[Icol,Jcol]=ind2sub(size(grayImage),locs);
% rowwise local maxima
grayImageT = grayImage';
[~,locs]=findpeaks(grayImageT(:),'MinPeakHeight', thres, 'MinPeakDistance', graphGroupingDistance);
[Jrow,Irow]=ind2sub(size(grayImageT),locs);


% pairwise distance
[intersectIdxCol,intersectDCol]= knnsearch([Icol,Jcol],[Irow,Jrow]);
[intersectIdxRow,intersectDRow]= knnsearch([Irow,Jrow],[Icol,Jcol]);
intersectIdxCol = intersectIdxCol(intersectDCol<=2);
intersectIdxRow = intersectIdxRow(intersectDRow<=2);

I = cat(1,Icol(intersectIdxCol),Irow(intersectIdxRow));
J = cat(1,Jcol(intersectIdxCol),Jrow(intersectIdxRow));

C = unique([I,J],'rows');
if isempty(C)
    p = [];
    return;
end
I = C(:,1);
J = C(:,2);

Ic = I;
Jc = J;

nPoints = length(Ic);
if nPoints > 1
    
    radius = round(graphGroupingDistance/2);
    % partition a space domain %%%%%%%
    nparts = ceil(size(Ic,1)/1000);
    idx = kmeans([Ic,Jc],nparts);
    Ictemp = [];
    Jctemp = [];
    for i = 1:nparts
        [I,J] = reduceMax(Ic(idx==i),Jc(idx==i),radius,grayImage);
        Ictemp = cat(1,Ictemp,I);
        Jctemp = cat(1,Jctemp,J);
    end
    Ic = Ictemp;
    Jc = Jctemp;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    nPoints = length(Ic);
    if nPoints > 1
        for radius = round(graphGroupingDistance/2)+1:graphGroupingDistance
            %  partition a space domain %%%%%%
            nparts = ceil(size(Ic,1)/1000);
            idx = kmeans([Ic,Jc],nparts);
            Ictemp = [];
            Jctemp = [];
            for i = 1:nparts
                [I,J] = reduceMax(Ic(idx==i),Jc(idx==i),radius,grayImage);
                Ictemp = cat(1,Ictemp,I);
                Jctemp = cat(1,Jctemp,J);
            end
            Ic = Ictemp;
            Jc = Jctemp;
            %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            if nPoints > length(Ic)
                nPoints = length(Ic);
                if nPoints <= 1
                    break;
                end
            else
                break;
            end
        end
    end
    
end
p = [Jc,Ic];
p = unique(p,'rows');
end

function [Ic,Jc] = reduceMax(I,J,graphGroupingDistance,grayImage)

C = [I,J];
distMat = squareform(pdist(C,'euclidean'));

dBinary = distMat < graphGroupingDistance;
clear distMat;

dBinary(logical(eye(size(dBinary)))) = 0;

m = ELSclique(dBinary);

nCliques = size(m,2);

Ic = zeros(nCliques,1);
Jc = zeros(nCliques,1);

for i=1:nCliques
    Im = I(logical(full(m(:,i))));
    Jm = J(logical(full(m(:,i))));
     
    weight = zeros(length(Im),1);
    for j = 1:length(Im)
        weight(j) = grayImage(Im(j),Jm(j));
    end
    
    [~,maxIdx] = max(weight);
    
    Ic(i) = Im(maxIdx);
    Jc(i) = Jm(maxIdx);
end

Ic = round(Ic);
Jc = round(Jc);
end