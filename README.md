# H&E-based single-cell identification pipeline
Deep learning pipeline repository for our paper "xxxx" published in xxx. The pipeline accepts a standard H&E (e.g. ndpi format) and outputs  The SCCNN method was first published in doi.org/10.1109/TMI.2016.2525803 but re-implemented in Python-TensorFlow here. Tissue segmentation is based on MicroNet: doi.org/10.1016/j.media.2018.12.003. 

## Citation
If you use this pipeline or some of its steps, please cite: 
* AbdulJabbar, K. et al 2020 

## Highlights 
<p align="center">
  <img width="800" src="https://github.com/qalid7/compath/blob/master/common/images/pipeline.png">
</p>

The steps can be further explained as follows: 

* Tiling: to convert a raw microscopy image into 2000x2000 tiles.  
* Tissue segmentation: 

The above two steps can be skipped, e.g. if you already have small sections of a H&E as JPEG tiles, or if you don't think there is any need to segment tissue areas. However, please note, tissue segmentation is a fast step that rids 

* Cell detection: 
* Cell classification: 

Both cell detection and classification algorithms contain pre processing routines. You can turn this off/on or modify it from: 

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
## Example data

Under data/example we provide a sample xx. This is to compare against your .

* example/data: 
* example/results:  

## Post processsing

A likely scenario is to see a lot of rubbish being detected outside the tissue regions. Simply because our algorithm . Though this should be taken care of using 


## Test data (LATTICe-A annotations) 
<p align="center">
  <img height="150" src="https://github.com/qalid7/compath/blob/master/common/images/ann_data.png">
</p>

Single-cell pathology annotations from the LATTICe-A cohort are provided under data. This test dataset represents one of several external validations performed in the paper.  

## Training 

We highly recommend using our more recent version (updated codes, tf version 1.13) of this pipeline if you wish to train models from scratch. This will be provided in a seprate [repo](xxxxxxxxxxxxxxx). 

## License 
