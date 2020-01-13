import h5py
import numpy as np


def h5read(filename, data_name):
    file = h5py.File(filename)
    data = file[data_name + '/data'].value
    shape = file[data_name + '/shape'].value
    data = data.reshape(shape, order='F')

    file.close()
    return data


def h5write(filename, data_to_save, name_data):
    file = h5py.File(filename, 'w')
    data = data_to_save.ravel(order='F')
    shape = np.array(data_to_save.shape)
    file.create_dataset(name_data + '/data', data.shape, dtype=data.dtype, data=data,
                        chunks=True, compression="gzip", compression_opts=5)
    file.create_dataset(name_data + '/shape', shape.shape, dtype=shape.dtype, data=shape,
                        chunks=True, compression="gzip", compression_opts=5)
    file.close()
