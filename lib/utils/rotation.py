import numpy as np
import torch


def rodriguez(vec1, vec2):
    '''
    calculate rotate matrix from norm vec1 to norm vec2 (numpy)
    '''
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    # print("vec1:", vec1)
    # print("vec2:", vec2)
    theta = np.arccos(vec1 @ vec2)

    vec_n = np.cross(vec1, vec2)

    n_1 = vec_n[0]
    n_2 = vec_n[1]
    n_3 = vec_n[2]
    vec_nx = np.asarray([[0, -n_3, n_2], [n_3, 0, -n_1], [-n_2, n_1, 0]])
    c = np.dot(vec1, vec2)
    s = np.linalg.norm(vec_n)
    if s == 0:
        return np.eye(3)
    R = np.eye(3) + vec_nx + vec_nx.dot(vec_nx) * ((1 - c) / (s**2))

    # print("theta:", theta)
    # print("n:", vec_nx)
    # print("R:", R)
    return R


def rodriguez_torch(vec1, vec2):
    '''
    calculate rotate matrix from norm vec1 to norm vec2 (torch tensor)
    '''
    vec1 = vec1.float()
    vec2 = vec2.float()
    vec1 = vec1 / torch.linalg.norm(vec1)
    vec2 = vec2 / torch.linalg.norm(vec2)
    # theta = torch.arccos(vec1 @ vec2)

    vec_n = torch.cross(vec1, vec2)
    n_1 = vec_n[0]
    n_2 = vec_n[1]
    n_3 = vec_n[2]
    vec_nx = torch.Tensor([[0, -n_3, n_2], [n_3, 0, -n_1],
                           [-n_2, n_1, 0]]).to(vec1.device)
    c = torch.dot(vec1, vec2)
    s = torch.linalg.norm(vec_n)
    if s.item() == 0:
        return torch.eye(3, device=vec1.device)
    # print("s:", s)
    R = torch.eye(3, device=vec1.device) + vec_nx + vec_nx @ vec_nx * (
        (1 - c) / (s**2))

    # print("theta:", theta)
    # print("n:", vec_nx)
    # print("R:", R)
    return R


