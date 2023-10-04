import train
from lib.utils import  vis, cfg
from lib.datasets import utils as dataset_utils
import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import cv2
import torch


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument("--index", type=int, required=True, help="Index, the idx of pose")
    parser.add_argument("--res_folder", type=str, required=True, help="Path, where keypoints results are stored")
    parser.add_argument('--method', type=str, default='ltri', help="Method used")
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--logdir", type=str, default="./logs", help="Path, where logs will be stored")
    parser.add_argument("--vis_type", type=str, default="val", help="dataset kinds")

    args = parser.parse_args()
    return args


def vis_index(data, indexs, keypoints_3d_pred_stereo, keypoints_3d_pred_ltrialg, keypoints_3d_pred_ltrivol, keypoints_3d_gt, mpjpes, config, vis_folder = './res/vis', dataset='h36m'):
    device = torch.device('cuda:0')
    keypoints_3d_i = 0
    vis_folder = os.path.join(vis_folder, dataset)
    os.makedirs(vis_folder, exist_ok=True)
    for iter_i, batch in enumerate(data):
        for batch_i, idx in enumerate(batch['indexes']):
            if idx in indexs:
                print("index:", idx)
                images_batch, _, _,_,_,_,_,_,_,_,_,_ = dataset_utils.prepare_batch(batch, device, config)
  
                # plot 2d
                keypoints_vis = vis.visualize_batch_cv2(
                    images_batch,None,
                        None,
                        None,
                        None,
                        None,
                    kind="human36m",
                    batch_index=batch_i, size=5,
                    max_n_cols=10
                )
                cv2.imwrite(os.path.join(vis_folder, "images_vis_{}_{}.png".format(dataset, idx)), keypoints_vis)
                print("Successfully draw 2d result of {}th pose.".format(idx))
                # plot 3d
                keypoints_3d_pred_stereo_numpy = keypoints_3d_pred_stereo[keypoints_3d_i]
                keypoints_3d_pred_ltrialg_numpy = keypoints_3d_pred_ltrialg[keypoints_3d_i]
                keypoints_3d_pred_ltrivol_numpy = keypoints_3d_pred_ltrivol[keypoints_3d_i]
                keypoints_3d_gt_numpy = keypoints_3d_gt[keypoints_3d_i]

                keypoints_3d_pred_stereo_numpy = keypoints_3d_pred_stereo_numpy - keypoints_3d_pred_stereo_numpy[6,:]
                keypoints_3d_pred_ltrialg_numpy = keypoints_3d_pred_ltrialg_numpy - keypoints_3d_pred_ltrivol_numpy[6,:]
                keypoints_3d_pred_ltrivol_numpy = keypoints_3d_pred_ltrivol_numpy - keypoints_3d_pred_ltrivol_numpy[6,:]
                keypoints_3d_gt_numpy = keypoints_3d_gt_numpy - keypoints_3d_gt_numpy[6,:]
               
                fig = plt.figure(figsize=(16,14), dpi=1200)
                axes = fig.subplots(nrows=3, ncols=4, subplot_kw=dict(fc='whitesmoke', projection='3d',))
                # print(keypoints_3d_gt_numpy)
                vis.draw_3d_pose_in_two_views(axes[0], keypoints_3d_gt=keypoints_3d_gt_numpy, keypoints_3d_pred=keypoints_3d_pred_stereo_numpy)
                vis.draw_3d_pose_in_two_views(axes[1], keypoints_3d_gt=keypoints_3d_gt_numpy, keypoints_3d_pred=keypoints_3d_pred_ltrialg_numpy)
                vis.draw_3d_pose_in_two_views(axes[2], keypoints_3d_gt=keypoints_3d_gt_numpy, keypoints_3d_pred=keypoints_3d_pred_ltrivol_numpy)
                plt.savefig(os.path.join(vis_folder, "keypoints_3d_vis_{}_{}_s{:.2f}_a{:.2f}_v{:.2f}.png".format(dataset, idx, mpjpes[keypoints_3d_i][0], mpjpes[keypoints_3d_i][1], mpjpes[keypoints_3d_i][2])))
                plt.close()
                
                keypoints_3d_i += 1


def main(args_path, path_stereo, path_ltrialg, path_ltrivol, path_gt, path_index):     
    # config
    config = cfg.load_config(args_path)
    # config.opt.batch_size = 1
    config.dataset.norm=False
    is_distributed = False

    # datasets
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = train.setup_dataloaders(config, distributed_train=is_distributed)

    keypoints_3d_pred_stereo = np.load(path_stereo)
    keypoints_3d_pred_ltrialg = np.load(path_ltrialg)
    keypoints_3d_pred_ltrivol = np.load(path_ltrivol)
    keypoints_3d_gt = np.load(path_gt)
    index = np.load(path_index)

    mpjpe_stereo = np.mean(np.sqrt(np.sum((keypoints_3d_pred_stereo - keypoints_3d_gt)**2, axis=2)), axis=1)
    mpjpe_ltrialg = np.mean(np.sqrt(np.sum((keypoints_3d_pred_ltrialg - keypoints_3d_gt)**2, axis=2)), axis=1)
    mpjpe_ltrivol = np.mean(np.sqrt(np.sum((keypoints_3d_pred_ltrivol - keypoints_3d_gt)**2, axis=2)), axis=1)

    indexs = []
    keypoints_3d_stereo=[]
    keypoints_3d_ltrialg=[]
    keypoints_3d_ltrivol=[]
    keypoints_3d_gt_vis=[]
    mpjpes = []
    datalen = mpjpe_stereo.shape[0]
    for l in range(datalen):
        if mpjpe_stereo[l] < (mpjpe_ltrialg[l]-20) and mpjpe_stereo[l] < (mpjpe_ltrivol[l]-15):
            indexs.append(index[l])
            mpjpes.append([mpjpe_stereo[l], mpjpe_ltrialg[l], mpjpe_ltrivol[l]])
            keypoints_3d_stereo.append(keypoints_3d_pred_stereo[l])
            keypoints_3d_ltrialg.append(keypoints_3d_pred_ltrialg[l])
            keypoints_3d_ltrivol.append(keypoints_3d_pred_ltrivol[l])
            keypoints_3d_gt_vis.append(keypoints_3d_gt[l])
            
    keypoints_3d_stereo = np.array(keypoints_3d_stereo)
    keypoints_3d_ltrialg = np.array(keypoints_3d_ltrialg)
    keypoints_3d_ltrivol = np.array(keypoints_3d_ltrivol)
    keypoints_3d_gt_vis = np.array(keypoints_3d_gt_vis)
    vis_index(val_dataloader, indexs, keypoints_3d_stereo, keypoints_3d_ltrialg, keypoints_3d_ltrivol, keypoints_3d_gt_vis, mpjpes, config, vis_folder = './vis/compare/', dataset='h36m')


if __name__ == '__main__':
    args_path = '/data0/wxy/3d_pose/stereo-estimation/experiments/mhad/train/mhad_stereo_volume_resnet152.yaml'
    path_stereo = "/data0/wxy/3d_pose/stereo-estimation/logs/eval_mhad_m_stereo_volume_152_StereoTriangulationNet@14.09.2023-15:32:49/keypoints_3d_pred.npy" 
    path_ltrialg = "/data0/wxy/3d_pose/learnable-triangulation-pytorch/res/keypoints_3d_alg_mhad.npy"
    path_ltrivol = "/data0/wxy/3d_pose/learnable-triangulation-pytorch/res/keypoints_3d_vol_mhad.npy" 
    path_gt = "/data0/wxy/3d_pose/stereo-estimation/logs/eval_mhad_m_stereo_volume_152_StereoTriangulationNet@14.09.2023-15:32:49/keypoints_3d_gt.npy" 
    path_index = "/data0/wxy/3d_pose/learnable-triangulation-pytorch/res/indexes_mhad.npy"
    main(args_path, path_stereo, path_ltrialg, path_ltrivol, path_gt, path_index)
