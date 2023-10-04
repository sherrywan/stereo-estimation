export CUDA_VISIBLE_DEVICES=1,2,3
python3 -m torch.distributed.launch --nproc_per_node=3 --master_port=2345 \
  train.py \
  --config ./experiments/h36m/train/h36m_stereo_volume_resnet50.yaml \
  --logdir ./logs