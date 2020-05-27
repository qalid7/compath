# H&E-based single-cell identification pipeline
Deep learning pipeline repository for our paper "Geospatial immune variability illuminates differential evolution of lung adenocarcinoma" published in Nature Medicine. 

In addition to a combination of Python, MATLAB and R scripts, this repository also includes, example H&E images and their final outputs to get you started and single-cell annotations data for external cohort testing. 

The pipeline accepts a standard H&E (e.g. ndpi format) and outputs a spatial map, where all cancer, lymphocyte and stromal cells can be recognized. The SCCNN method was first published in doi.org/10.1109/TMI.2016.2525803 but re-implemented in Python-TensorFlow here. Tissue segmentation is based on MicroNet: doi.org/10.1016/j.media.2018.12.003. 

## Citation
If you use this pipeline or some of its steps, or if you use the attached annotation data or checkpoint files, please cite: 
* AbdulJabbar, K. et al. Geospatial immune variability illuminates differential evolution of lung adenocarcinoma. Nature Medicine (2020). doi: 10.1038/s41591-020-0900-x

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

To execute, you need the below Conda virtual environments. 

## Python-TensorFlow virtual envs (Linux) 

* For cell detection and classification: 

```
module load anaconda/3/4.4.0
conda create -n tfdavrosCPU1p3 python=3.5.4
conda activate tfdavrosCPU1p3
conda install scipy=0.19 pandas=0.20 numpy=1.13.1
pip install /apps/tensorflow/tensorflow-1.3.0-cp35-cp35m-linux_x86_64.whl

cd /apps/MATLAB/R2018b/extern/engines/python
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

* latticea_test_data/gt_celllabels: expert pathology annotations in the form of class, and x,y coordinates. 

* latticea_test_data/dl_celllabels: our final cell predictions from this pipeline.


## Multiplex IHC

By large, this pipeline is designed for H&E images as they make the bulk of our paper. For multiplex IHC images (CD8+, CD4+FOXP3-, CD4+FOXP3+; refer to Methods in the paper). Depending on your IHC images (combination of colors, cytoplasmic/nuclear staining), the pipeline may need some modification. 

## Training 

Training codes are available for each step of this pipeline. We will update this repo with a more recent version (updated codes, tf version 1.13) of this pipeline. 
