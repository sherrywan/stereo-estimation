'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-05-15 14:37:13
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-05-16 13:42:45
FilePath: /wxy/3d_pose/stereo-estimation/lib/datasets/mhad_preprocess/generate-pairwise_attention.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import json
import torch
import torch.nn.functional as F
import os
import numpy as np


def generate_pair_attention(json_datas, volume_size, cuboid_size, temperature, save_tag = False, save_dir='/data0/wxy/3d_pose/stereo-estimation/limb_volume_data'):
    # generate grid coordinate
    xx, yy, zz = torch.meshgrid(torch.arange(volume_size[0]), torch.arange(volume_size[1]), torch.arange(volume_size[2]))
    #(d, h, w)
    grid_length = torch.tensor([cuboid_size / (w - 1) for w in volume_size])
    grid_coordinate = torch.cat((xx.unsqueeze(0) * grid_length[0],\
        yy.unsqueeze(0) * grid_length[1], zz.unsqueeze(0) * grid_length[2]), dim = 0).cuda().float() #(3, d, h, w)
    grid_coordinate = grid_coordinate.view(3, -1).permute(1, 0) #(d*h*w, 3)
    
    grid_delta = grid_coordinate.unsqueeze(1) - grid_coordinate.unsqueeze(0) #(d*h*w, d*h*w, 3)
    grid_dis = torch.norm(grid_delta, dim = 2) #(d*h*w, d*h*w)

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
            pair_attention_grid = F.softmax(-torch.pow(grid_dis - mean, 2) / (temperature * 2 * torch.pow(std, 2) + 1e-1), dim = 1).unsqueeze(0)
            if save_tag:
                save_path = os.path.join(save_dir, f"{name}_{child}_pairattention_temp{temperature}.npy")
                pair_attention = pair_attention_grid.detach().cpu().numpy()
                np.save(save_path, pair_attention)
                

if __name__ == "__main__":
    json_file = "/data1/share/dataset/MHAD_Berkeley/stereo_camera/extra/human36m-stereo-m-bones_train.json"
    volume_size=[16,16,16]
    cuboid_size=2500.0
    temperatures = [1,10,100]
    with open(json_file, 'r') as f:
        datas = json.load(f)
        datas = eval(datas)
    for temperature in temperatures:
        generate_pair_attention(datas, volume_size, cuboid_size, temperature, save_tag = True, save_dir='../../../limb_volume_data')
    
