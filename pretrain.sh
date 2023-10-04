export CUDA_VISIBLE_DEVICES=0
python3 pretrain_stepMaskST.py \
  --config ./experiments/mhad/train/mhad_pretrain_MaskST.yaml \
  --logdir ./logs