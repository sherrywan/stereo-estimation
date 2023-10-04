# MHAD_Berkeley stereo dataset preprocess

## download dataset and preprocess it  
according to https://github.com/sherrywan/MHAD_Berkeley_preprocess

## generat labels  
run 
```
bash generate-labels.sh 
```
to generate labels.npy and recmap.npy

## rectificate stereo images  
run 
```
bash rectificate.sh 
```
to rectificate stereo images

## check labels
run
```
bash check.sh
```
to check the GT labels and dataset class