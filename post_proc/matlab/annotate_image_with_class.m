function image = annotate_image_with_class(image, points, colour, strength)
    
    if size(image, 3)~=3
        error('Please input an RGB image');
    end
    image = im2double(image);
    label = zeros(size(image,1), size(image,2));
    linearInd = sub2ind(size(label), points(:,2), points(:,1));
    label(linearInd) = 1;
    label = imdilate(label, strel('disk', strength))>0;
%     label = logical(repmat(label, [1, 1, 3]));
    image1 = image(:,:,1);
    image2 = image(:,:,2);
    image3 = image(:,:,3);
    image1 = image1(:);
    image2 = image2(:);
    image3 = image3(:);
    image1(label(:)) = colour(1);
    image2(label(:)) = colour(2);
    image3(label(:)) = colour(3);
    image1 = reshape(image1, size(label));
    image2 = reshape(image2, size(label));
    image3 = reshape(image3, size(label));
    image = cat(3, image1, image2, image3);
end