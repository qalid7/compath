#!/bin/bash
#VERSION v0.1

#Find NDPI etc and extract them

#$1 Source directory
#$2 Target

# bash CellDetect.sh `pwd`
# find . -name "*.bsubHEdetection" -exec bash -c "bsub < {}" \;

SAMPLES=`find ${1} -type d -name "*.ndpi"`
FULLDIRNAME=${1}
mkdir -p "${1}/bsubHEdetection"
mkdir -p "${1}/errorsHEdetection"
mkdir -p "${1}/outputsHEdetection"

#cp /mnt/scratch/users/molecpath/sraza/scripts/MyCodes/TracerX/detection/20171013SCCNN_detection/parameters-detection.txt "${1}"
FULLDIRNAMEESC=$(echo $FULLDIRNAME | sed 's_/_\\/_g') 
for s in $SAMPLES; do	
  echo "Create bsub for ${s}"
  BNAME=`basename ${s}`
  cp /mnt/scratch/users/molecpath/sraza/scripts/MyCodes/TracerX/detection/20171013SCCNN_detection/header_cell_detection "${FULLDIRNAME}/bsubHEdetection/${BNAME}_extract.bsubHEdetection"  
  sed -i "s/###NAME###/${BNAME}/g" "${FULLDIRNAME}/bsubHEdetection/${BNAME}_extract.bsubHEdetection"
  sed -i "s/###DIRNAME###/${FULLDIRNAMEESC}/g" "${FULLDIRNAME}/bsubHEdetection/${BNAME}_extract.bsubHEdetection"
done	

