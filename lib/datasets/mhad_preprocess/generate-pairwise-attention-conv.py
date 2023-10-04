'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-05-29 13:55:40
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-06-14 22:25:36
FilePath: /wxy/3d_pose/stereo-estimation/lib/datasets/mhad_preprocess/generate-pairwise-attention-conv.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import json
import torch
import torch.nn.functional as F
import os
import numpy as np


def generate_pair_attention(json_datas, volume_size, cuboid_size, temperature, save_tag = False, save_dir='/data0/wxy/3d_pose/stereo-estimation/limb_volume_data'):
    # generate grid coordinate
    x, y, z = torch.meshgrid(torch.arange(volume_size[0]), torch.arange(volume_size[1]), torch.arange(volume_size[2]))
    #(d, h, w)
    sides = torch.from_numpy(np.array(
                [cuboid_size, cuboid_size, cuboid_size]))
    position = - sides / 2
    grid_3d = torch.stack([x, y, z], dim=-1).type(torch.float)
    grid_3d = grid_3d.reshape((-1, 3))
            
    grid_coord = torch.zeros_like(grid_3d)
    grid_coord[:, 0] = position[0] + (sides[0] /
                                    (volume_size[0] - 1)) * grid_3d[:, 0]
    grid_coord[:, 1] = position[1] + (sides[1] /
                                    (volume_size[1] - 1)) * grid_3d[:, 1]
    grid_coord[:, 2] = position[2] + (sides[2] /
                                    (volume_size[2] - 1)) * grid_3d[:, 2]
    
    position_center = torch.tensor([0,0,0]).reshape(1,3)
    grid_dis = torch.sqrt(torch.sum((grid_coord - position_center) ** 2, dim= 1)).cuda().float() #(d*h*w,1)


    for data in json_datas:
        idx = data['idx']
        name = data['name']
        childs = data['children']
        means = data['bones_mean']
        stds = data['bones_std']
        means = torch.from_numpy(np.asarray(means)).cuda().float()
        stds = torch.from_numpy(np.asarray(stds)).cuda().float()
        for i,child in enumerate(childs):
            mean = means[i]
            std = stds[i]
            pair_attention = torch.exp(-torch.pow(grid_dis - mean, 2) / (temperature * 2 * torch.pow(std, 2)))
            pair_attention_grid = pair_attention.reshape(*volume_size)
            # pair_attention_grid = F.softmax(pair_attention, dim = 0).reshape(*volume_size)
            if save_tag:
                save_path = os.path.join(save_dir, f"{name}_{child}_pairattention_conv_temp{temperature}_wosoftmax.npy")
                pair_attention = pair_attention_grid.detach().cpu().numpy()
                np.save(save_path, pair_attention)
                

if __name__ == "__main__":
    json_file = "/data1/share/dataset/MHAD_Berkeley/stereo_camera/extra/human36m-stereo-m-bones_train.json"
    volume_size=[33,33,33]
    cuboid_size=1120.0
    temperatures = [1,2,5,10]
    with open(json_file, 'r') as f:
        datas = json.load(f)
        datas = eval(datas)
    for temperature in temperatures:
        generate_pair_attention(datas, volume_size, cuboid_size, temperature, save_tag = True, save_dir='../../../limb_volume_data')
    
