'''
Author: sherrywaan sherrywaan@outlook.com
Date: 2023-05-08 22:43:13
LastEditors: sherrywaan sherrywaan@outlook.com
LastEditTime: 2023-06-13 14:57:46
FilePath: /wxy/3d_pose/stereo-estimation/lib/utils/transform.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
import torch
import math

def space_trans_to_stereo(K_l, K_r, T_l, T_r, grid_coord, gt_2d_l, gt_2d_r):
    ''' trans 3D position (x,y,z) in left camera coordinate system
      to stereo position (u,v,d) in left image coordinate system
    
    Parameters:
        K_l : left camera intrinsics with shape (3,4)
        K_r : right camera intrinsics with shape (3,4)
        T_l : left camera translation with shape (3,)
        T_r : right camera translation with shape (3,)
        grid_coord : 3D coordinates volume with shape (volume_size, volume_size, volume_size, 3)

    Output:
        grid_coord_s : stereo coordinates volume with shape (volume_size, volume_size, volume_size, 3)
    '''

    f_xl = K_l[0,0]
    f_yl = K_l[1,1]
    t_xl = K_l[0,2]
    t_yl = K_l[1,2]
    s_l = K_l[0,1]

    f_xr = K_r[0,0]
    t_xr = K_r[0,2]

    baseline = torch.sqrt(torch.sum((T_l-T_r)**2))

    grid_coord_s = torch.zeros_like(grid_coord)

    grid_coord[:,2] = grid_coord[:,2] + 1e-9
    xz_ratio = grid_coord[:,0] / grid_coord[:,2]
    yz_ratio = grid_coord[:,1] / grid_coord[:,2]
    u_l = f_xl * xz_ratio + t_xl + s_l * yz_ratio
    v_l = f_yl * yz_ratio + t_yl

    u_r  = f_xr * xz_ratio - f_xr * baseline / grid_coord[:, 2] + f_xr / f_xl * s_l * yz_ratio + t_xr
    # u_l_delta = u_l - t_xl
    # d = f_xl * baseline / grid_coord[:,2]
    # u_r_delta = (d - u_l_delta) * f_xr / f_xl

    grid_coord_s[:, 0] = u_l
    grid_coord_s[:, 1] = v_l
    grid_coord_s[:, 2] = u_l - u_r

    return grid_coord_s


# 计算两点之间线段的距离
def __line_magnitude(x1, y1, x2, y2):
    lineMagnitude = math.sqrt(math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2))
    return lineMagnitude
 
 
def point_to_line_distance(point, line):
    px, py = point
    x1, y1, x2, y2 = line
    line_magnitude = __line_magnitude(x1, y1, x2, y2)
    if line_magnitude < 0.00000001:
        return 9999
    else:
        u1 = (((px - x1) * (x2 - x1)) + ((py - y1) * (y2 - y1)))
        u = u1 / line_magnitude
        if (u < 0.00001) or (u > 1):
            # 点到直线的投影不在线段内, 计算点到两个端点距离的最小值即为"点到线段最小距离"
            ix = __line_magnitude(px, py, x1, y1)
            iy = __line_magnitude(px, py, x2, y2)
            if ix > iy:
                distance = iy
            else:
                distance = ix
        else:
            # 投影点在线段内部, 计算方式同点到直线距离, u 为投影点距离x1在x1x2上的比例, 以此计算出投影点的坐标
            ix = x1 + u * (x2 - x1)
            iy = y1 + u * (y2 - y1)
            distance = __line_magnitude(px, py, ix, iy)
        return distance
