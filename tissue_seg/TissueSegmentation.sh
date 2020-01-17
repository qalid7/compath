#!/bin/bash
#VERSION v0.1

#Find NDPI etc and extract them

#$1 Source directory
#$2 Target

# bash TissueSegmentation.sh `pwd`
# find . -name "*.bsub_tissuesegment" -exec bash -c "bsub < {}" \;

SAMPLES=`find ${1} -type d -name "LTX*.ndpi"`
FULLDIRNAME=${1}
mkdir -p "${1}/bsub_tissuesegment"
mkdir -p "${1}/errors_tissuesegment"
mkdir -p "${1}/outputs_tissuesegment"

cp /mnt/scratch/users/molecpath/sraza/scripts/MyCodes/TracerX/tissue_segmentation/20171110-TissueSegmentation/parameters-tissuesegment.txt "${1}"
FULLDIRNAMEESC=$(echo $FULLDIRNAME | sed 's_/_\\/_g') 
for s in $SAMPLES; do	
  echo "Create bsub for ${s}"
  BNAME=`basename ${s}`
  cp /mnt/scratch/users/molecpath/sraza/scripts/MyCodes/TracerX/tissue_segmentation/20171110-TissueSegmentation/header_tissue_segmentation "${FULLDIRNAME}/bsub_tissuesegment/${BNAME}_extract.bsub_tissuesegment"  
  sed -i "s/###NAME###/${BNAME}/g" "${FULLDIRNAME}/bsub_tissuesegment/${BNAME}_extract.bsub_tissuesegment"
  sed -i "s/###DIRNAME###/${FULLDIRNAMEESC}/g" "${FULLDIRNAME}/bsub_tissuesegment/${BNAME}_extract.bsub_tissuesegment"
done	

