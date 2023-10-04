# export CUDA_VISIBLE_DEVICES=1,2,3
python3 infer_ST_withinitial3D.py \
  --config ./experiments/mhad/eval/mhad_refine_ST.yaml \
  --logdir ./logs