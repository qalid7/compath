# H&E-based single-cell identification pipeline
Deep learning pipeline repository for our paper "xxxx" published in xxx. The pipeline accepts a standard H&E (e.g. ndpi format) and outputs  The SCCNN method was first published in 10.1109/TMI.2016.2525803 but re-implemented in Python-TensorFlow here. Tissue segmentation doi.org/10.1016/j.media.2018.12.003

## Trained models

Trained models (checkpoint files) can be downloaded from [here](https://www.dropbox.com/sh/98qaunytnm7u2zo/AABO08G1gvT9jz7KDhiB5BO9a?dl=0). You need to copy each 'checkpoint' folder from the dropbox link to the respective folder in this repositry (e.g. cell_class, etc). 


## Python-TensorFlow virtual envs (linux/cluster) 

* For cell detection and classification: 

```
module load anaconda/3/4.4.0
conda create -n tfdavrosCPU1p3 python=3.5.4
conda activate tfdavrosCPU1p3
conda install scipy=0.19 pandas=0.20 numpy=1.13.1

cd /apps/MATLAB/R2017b/extern/engines/python
#replace your dir:
python setup.py build --build-base="/home/dir/tmp" install
pip install pillow==4.2.1 h5py==2.7.1
conda deactivate

#check by running python then 'import tensorflow as tf'
```
* For tiling raw ndpi files: 

```
module load anaconda/3/4.4.0
conda create –n CWS python=3.5
source activate CWS
conda install numpy
module load java/sun8/1.8.0u66
pip install 'python-bioformats<=1.3.0'
module load openjpeg/2.1.2
module load openslide/3.4.1
pip install openslide-python
source deactivate CWS
```
## Test data (LATTICe-A annotations) 
<p align="center">
  <img width="600" height="200" src="https://github.com/qalid7/compath/blob/master/common/images/ann_data.png">
</p>

## Training 

We highly recommend using our more recent versions (updated codes, tf 1.13) of this pipeline to retrain data from scratch. This is provided in a seprate [repo](xxxxxxxxxxxxxxx). 
