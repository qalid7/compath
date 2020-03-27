function [] = h5save(saveName, data, name)

data_set_name = sprintf('/%s/data', name);
shape_set_name = sprintf('/%s/shape', name);

h5create(saveName, data_set_name, ...
    size(data(:)),'ChunkSize',size(data(:)),'Deflate',5, 'Datatype', class(data))
h5create(saveName, shape_set_name, ...
    length(size(data)),'ChunkSize',length(size(data)),'Deflate',5, ...
    'Datatype', 'int64')
h5write(saveName, data_set_name, data(:));
h5write(saveName, shape_set_name, int64(size(data)));
end