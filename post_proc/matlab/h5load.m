function data = h5load(filename, name)

data_set_name = sprintf('/%s/data', name);
shape_set_name = sprintf('/%s/shape', name);
raw_data = h5read(filename,data_set_name);
shape = h5read(filename,shape_set_name);
data = reshape(raw_data, shape');

end