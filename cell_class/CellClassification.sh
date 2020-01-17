#!/bin/bash
#VERSION v0.1

#Find NDPI etc and extract them

#$1 Source directory
#$2 Target

# bash CellClassification.sh `pwd`
# find . -name "*.bsubclassification" -exec bash -c "bsub < {}" \;

SAMPLES=`find ${1} -type d -name "*.ndpi"`
FULLDIRNAME=${1}
mkdir -p "${1}/bsubclassification"
mkdir -p "${1}/errorsclassification"
mkdir -p "${1}/outputsclassification"

cp /mnt/scratch/users/molecpath/sraza/scripts/MyCodes/TracerX/classification/20171019-SCCNNClassifier/parameters-classification.txt "${1}"
FULLDIRNAMEESC=$(echo $FULLDIRNAME | sed 's_/_\\/_g') 
for s in $SAMPLES; do	
  echo "Create bsub for ${s}"
  BNAME=`basename ${s}`
  cp /mnt/scratch/users/molecpath/sraza/scripts/MyCodes/TracerX/classification/20171019-SCCNNClassifier/header_cell_classification "${FULLDIRNAME}/bsubclassification/${BNAME}_extract.bsubclassification"  
  sed -i "s/###NAME###/${BNAME}/g" "${FULLDIRNAME}/bsubclassification/${BNAME}_extract.bsubclassification"
  sed -i "s/###DIRNAME###/${FULLDIRNAMEESC}/g" "${FULLDIRNAME}/bsubclassification/${BNAME}_extract.bsubclassification"
done	

