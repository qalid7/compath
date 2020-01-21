# H&E-based single-cell identification pipeline
Deep learning pipeline repository for our paper "xxxx" published in xxx. The pipeline accepts a standard H&E (e.g. ndpi format) and outputs  The SCCNN method was first published in 10.1109/TMI.2016.2525803 but re-implemented in Python-TensorFlow here. Tissue segmentation doi.org/10.1016/j.media.2018.12.003

* Trained models

Trained models (checkpoint files) can be downloaded from [here] (https://www.dropbox.com/sh/98qaunytnm7u2zo/AABO08G1gvT9jz7KDhiB5BO9a?dl=0). 


* Python env (linux/cluster): 

⋅⋅⋅For cell detection and classification: 

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
⋅⋅⋅ For tiling raw ndpi files: 

```
xxx
```
