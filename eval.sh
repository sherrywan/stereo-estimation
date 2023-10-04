export CUDA_VISIBLE_DEVICES=0 
python3 train.py \
  --eval --eval_dataset val \
  --config ./experiments/mhad/eval/mhad_stereo_volume_resnet152.yaml \
  --logdir ./logs