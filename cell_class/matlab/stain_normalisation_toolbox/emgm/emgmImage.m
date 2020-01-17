function binImage = emgmImage( Image )
%EMGMIMAGE Summary of this function goes here
%   Detailed explanation goes here

    [label, ~, ~] = emgm(convertimage(Image), 5);
    binImage = reshape(label, size(Image, 1), size(Image, 2));
end

function obs = convertimage( image )
%CONVERTIMAGE Summary of this function goes here
%   Detailed explanation goes here
    image1 = rgb2hsv(image);
    image1 = cat(3, image1, image);
    %image1 = image;
    
    %image1 = cat(3, image1, (image1(:,:,1) - image1(:,:,2) + image1(:,:,3)));
    
    obs = zeros(size(image1, 3), numel(image1(:,:,1)));
    
    for i=1:size(image1, 3)
        channel = image1(:, :, i);
        obs(i, :) = channel(:);
    end
end