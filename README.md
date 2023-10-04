# stereo-estimation
estimate the 3D human pose from binocular images

***

## data preprocess

#### Human3.6
refer to <https://github.com/karfly/learnable-triangulation-pytorch/blob/master/mvn/datasets/human36m_preprocessing/README.md> step 1-5
and then
    cd ./lib/datasets/h36m_preprocess
    bash generate-labels.sh

#### MHAD
refer to [mhad readme](https://github.com/sherrywan/stereo-estimation/blob/main/lib/datasets/mhad_preprocess/readme.md)

***

## train

change the file_path in train.sh, and run:
    bash train.sh

## eval

change the file_path in eval.sh, and run:
    bash eval.sh


