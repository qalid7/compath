# H&E-based single-cell identification pipeline
Deep learning pipeline repository for our paper "xxxxxxxxxxxxxxxx" published in xxxxxxxxxxxxxxxxxxxxx. The pipeline accepts a standard H&E (e.g. ndpi format) and outputs  The SCCNN method was first published in doi.org/10.1109/TMI.2016.2525803 but re-implemented in Python-TensorFlow here. Tissue segmentation is based on MicroNet: doi.org/10.1016/j.media.2018.12.003. 

## Citation
If you use this pipeline or some of its steps, please cite: 
* xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

## Highlights 
<p align="center">
  <img width="800" src="https://github.com/qalid7/compath/blob/master/common/images/pipeline.png">
</p>

The steps can be further explained as follows: 

* Tiling: to convert a raw microscopy image into 2000x2000 JPEG tiles.  
* Tissue segmentation: to segment viable tissue area from a H&E slide. 

The above two steps can be skipped, e.g. if you already have small sections of a H&E as JPEG tiles, or if you don't think there is any need to segment tissue areas. However, please note, tissue segmentation is a fast step that rids large unwanted tiles from a standard H&E to save time for the next two steps. 

* Cell detection: identifying cell nucleus, 
* Cell classification: predicting the class of an identified cell (cancer, stromal, lymphocyte, other)

Both cell detection and classification algorithms contain pre processing routines. You can turn this off/on or modify it from the main run script or sub matlab dir.  

## Trained models

Trained models (checkpoint files) can be downloaded from [here](https://www.dropbox.com/sh/98qaunytnm7u2zo/AABO08G1gvT9jz7KDhiB5BO9a?dl=0). You need to copy each 'checkpoint' folder from the dropbox link to the corrosponding folder in this repositry (e.g. cell_class). 


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

Under data/example we provide sample tiles. This is to get you started. The aim should be to run both cell detection and classification and replicate the results as seen under example/results. 

* example/data: raw tiled JPEGs, ready for cell detection and cell classification.
* example/results: the output of this pipeline in the form of annotated images and cell coordinates. 

## Post processsing

A likely scenario is to see a lot of rubbish being detected outside the tissue regions. This happens simply because our algorithm hasn't seen enough 'negative non-cell' events from a chohort other than Lung TRACERx. Though much of this rubbish should be avoided with tissue segmentation, however, we provide a simple MATLAB script for post processing (cleaning) under: post_proc. This script should also create a summary for all slides in one table: number and relative percentage of cells identified for each class. 


## Test data (LATTICe-A annotations) 
<p align="center">
  <img height="150" src="https://github.com/qalid7/compath/blob/master/common/images/ann_data.png">
</p>

Single-cell expert pathology annotations from the LATTICe-A cohort are provided under: test_data. This test dataset represents one of several external validations performed in the paper. 

The R scripts is provided to re-generate single-cell accuracy results - you should be able to replicate Table S3 from the paper using:    

* latticea_test_data/imgs: the original raw H&E tiles used for single-cell pathology annoations.

* latticea_test_data/gt_celllabels: expert pathology annotations in the form of x,y coordinates. 
* latticea_test_data/gt_annotated: pathology annotations visualised on the images (for colors look at Fig. 1 in the paper). 

* latticea_test_data/dl_celllabels: our final cell predictions from this pipeline, you should be able to get to these results using this repo and the 'imgs' data. 


## Multiplex IHC

By large, this pipeline is designed for H&E images as they make the bulk of our paper. We do provide checkpoint files for all steps in this pipeline trained for multiplex IHC images (CD8+, CD4+FOXP3-, CD4+FOXP3+; refer to Methods in the paper). However, depending on your IHC images (combination of colors, cytoplasmic/nuclear staining), the pipeline may need some modification or even a fresh training.  

## Training 

Training codes are available for each step if you wish to train models from scratch. However, we highly recommend using our more recent version (updated codes, tf version 1.13) of this pipeline provided in a seprate [repo](xxxxxxxxxxxxxxx). 

## License 
