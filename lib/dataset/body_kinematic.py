import joblib
from matplotlib.pyplot import thetagrids
import numpy as np
from numpy.core.numeric import NaN
from torch.jit import Error
from lib.utils.op import rodriguez, rodriguez_torch
import matplotlib.pyplot as plt
from lib.utils.vis import point3d
import os
import torch
from math import pi
from lib.utils import gmm


class Body_Kinematic:
    def __init__(self, keypoints, save_folder='./'):
        self.index = 0
        self.save_folder = save_folder
        self.keypoints = keypoints
        self.limb = [[0, 1], [1, 2], [2, 6], [5, 4], [4, 3], [3, 6], [6, 7],
                     [7, 8], [8, 16], [16, 9], [12, 8], [11, 12], [10, 11],
                     [13, 8], [14, 13], [15, 14]]
        self.H36M_NAMES = [''] * 16
        self.H36M_NAMES[0] = 'RFoot'
        self.H36M_NAMES[1] = 'RKnee'
        self.H36M_NAMES[2] = 'RHip'
        self.H36M_NAMES[3] = 'LFoot'
        self.H36M_NAMES[4] = 'LKnee'
        self.H36M_NAMES[5] = 'LHip'
        self.H36M_NAMES[6] = 'Spine'
        self.H36M_NAMES[7] = 'Thorax'
        self.H36M_NAMES[8] = 'Neck/Nose'
        self.H36M_NAMES[9] = 'Head'
        self.H36M_NAMES[10] = 'RShoulder'
        self.H36M_NAMES[11] = 'RElbow'
        self.H36M_NAMES[12] = 'RWrist'
        self.H36M_NAMES[13] = 'LShoulder'
        self.H36M_NAMES[14] = 'LElbow'
        self.H36M_NAMES[15] = 'LWrist'

        self.joint_angle = {
            12: {
                'name': 'RSho',
                'parent': 8,
                'child': 11,
                'gmm_score': 0.04,
                'u': [1, 0, 0],
                'v': [0, 0, -1],
                'z': [0, 1, 0]
            },
            11: {
                'name': 'RElb',
                'parent': 12,
                'child': 10,
                'gmm_score': 0.10,
                'u': [1, 0, 0],
                'v': [0, 0, -1],
                'z': [0, 1, 0]
            },
            13: {
                'name': 'LSho',
                'parent': 8,
                'child': 14,
                'gmm_score': 0.04,
                'u': [-1, 0, 0],
                'v': [0, 0, -1],
                'z': [0, -1, 0]
            },
            14: {
                'name': 'LElb',
                'parent': 13,
                'child': 15,
                'gmm_score': 0.12,
                'u': [-1, 0, 0],
                'v': [0, 0, -1],
                'z': [0, -1, 0]
            },
            2: {
                'name': 'RHip',
                'parent': 6,
                'child': 1,
                'gmm_score': 0.14,
                'u': [1, 0, 0],
                'v': [0, 0, -1],
                'z': [0, 1, 0]
            },
            1: {
                'name': 'RKne',
                'parent': 2,
                'child': 0,
                'gmm_score': 0.09,
                'u': [0, 0, -1],
                'v': [1, 0, 0],
                'z': [0, -1, 0]
            },
            3: {
                'name': 'LHip',
                'parent': 6,
                'child': 4,
                'gmm_score': 0.019,
                'u': [-1, 0, 0],
                'v': [0, 0, -1],
                'z': [0, -1, 0]
            },
            4: {
                'name': 'LKne',
                'parent': 3,
                'child': 5,
                'gmm_score': 0.012,
                'u': [0, 0, -1],
                'v': [-1, 0, 0],
                'z': [0, 1, 0]
            },
            7: {
                'name': 'Spine',
                'parent': 6,
                'child': 8,
                'gmm_score': 0.04,
                'u': [0, 0, 1],
                'v': [1, 0, 0],
                'z': [0, 1, 0]
            }
        }

        self.PI = 3.141592653589793

    def get_H3dM_NAMES(self):
        return self.H36M_NAMES

    def keypoints2bone(self):
        '''
        tranform keypoints to kcs bone vector according to self.limb relationship
        '''
        sample_num = self.keypoints.shape[0]
        keypoints = self.keypoints.reshape(sample_num, -1)
        limb_len = len(self.limb)
        # kcs trans matrix for vector whose shape is (N, 17*3)
        C = np.zeros((limb_len * 3, 17 * 3))
        for i in range(limb_len):
            j = self.limb[i]
            C[3 * i, 3 * j[0]] = 1
            C[3 * i, 3 * j[1]] = -1
            C[3 * i + 1, 3 * j[0] + 1] = 1
            C[3 * i + 1, 3 * j[1] + 1] = -1
            C[3 * i + 2, 3 * j[0] + 2] = 1
            C[3 * i + 2, 3 * j[1] + 2] = -1

        # print("kcs trans matrix:", C)
        bl_vector = np.einsum('jk,ik -> ij', C, keypoints)
        bl_vector = bl_vector.reshape(sample_num, limb_len, -1)
        return bl_vector

    def keypoints2angle(self, keypoints):
        '''
        tranform keypoints to joint angles in spherical coordinate
        input:
            keypoints: numpy (N,3) joints locations of one person
        '''
        angle_list = self.joint_angle
        angle_nums = len(angle_list)
        angle_res = np.zeros((angle_nums, 2))
        # 判断左右关系修正u_t v_t
        is_positive = True
        if keypoints[12, 0] - keypoints[13, 0] < 0:
            is_positive = False
        for i, key in enumerate(angle_list):
            joint = keypoints[key, :]
            parent = keypoints[angle_list[key]['parent'], :]
            child = keypoints[angle_list[key]['child'], :]
            u_t = angle_list[key]['u'].copy()
            v_t = angle_list[key]['v'].copy()
            z_t = angle_list[key]['z'].copy()
            if not is_positive:
                u_t[0] = -u_t[0]
                v_t[0] = -v_t[0]

            # calculate the axies of parents
            u = joint - parent
            u_norm = np.linalg.norm(u)
            u = u / u_norm
            R = rodriguez(u_t, u)
            v = R @ v_t
            v_norm = np.linalg.norm(v)
            v = v / v_norm
            z = np.cross(u, v)
            z_norm = np.linalg.norm(z)
            z = z / z_norm

            # calculate the angle in spherical coordinate
            b = child - joint
            b_norm = np.linalg.norm(b)
            b_uv = b - np.dot(z, b) * z
            b_uv_norm = np.linalg.norm(b_uv)
            x_theta = np.dot(z, b) / (b_norm)
            x_theta = max(-1, x_theta)
            x_theta = min(1, x_theta)
            theta = np.arccos(x_theta)
            if b_uv_norm == 0:
                phi = 0
            else:
                x_phi = np.dot(u, b_uv) / (b_uv_norm)
                x_phi = max(-1, x_phi)
                x_phi = min(1, x_phi)
                phi = np.arccos(x_phi)

            theta = 180 * (theta / self.PI)
            phi = 180 * (phi / self.PI)
            vec_dir = np.cross(u, b)
            if np.dot(vec_dir, z) < 0:
                phi = 360 - phi
            angle_res[i, 0] = np.round(theta)
            angle_res[i, 1] = np.round(phi)
            vis = False
            if vis and key == 11 and theta > 90:
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                point3d(ax, 0, 20, keypoints, 'green')
                ax.plot((keypoints[11, 0], keypoints[11, 0] + 100 * u[0]),
                        (keypoints[11, 1], keypoints[11, 1] + 100 * u[1]),
                        (keypoints[11, 2], keypoints[11, 2] + 100 * u[2]),
                        color='black')
                ax.plot((keypoints[11, 0], keypoints[11, 0] + b_uv[0]),
                        (keypoints[11, 1], keypoints[11, 1] + b_uv[1]),
                        (keypoints[11, 2], keypoints[11, 2] + b_uv[2]),
                        color='yellow',
                        linestyle='--')
                ax.plot((keypoints[11, 0], keypoints[11, 0] + 100 * v[0]),
                        (keypoints[11, 1], keypoints[11, 1] + 100 * v[1]),
                        (keypoints[11, 2], keypoints[11, 2] + 100 * v[2]),
                        color='red')
                ax.plot((keypoints[11, 0], keypoints[11, 0] + 100 * z[0]),
                        (keypoints[11, 1], keypoints[11, 1] + 100 * z[1]),
                        (keypoints[11, 2], keypoints[11, 2] + 100 * z[2]),
                        color='purple')

                plt.close()
                self.index += 1
        return angle_res

    def keypoints2angle_torch(self, keypoints):
        '''
        tranform keypoints to joint angles in spherical coordinate
        input:
            keypoints: torch tensor (N,3) joints locations of one person
        '''
        device = keypoints.device
        angle_list = self.joint_angle
        angle_nums = len(angle_list)
        angle_res = torch.zeros(angle_nums, 2, device=device)
        # 判断左右关系修正u_t v_t
        is_positive = True
        if (keypoints[12, 0] - keypoints[13, 0]).item() < 0:
            is_positive = False
        for i, key in enumerate(angle_list):
            joint = keypoints[key, :]
            parent = keypoints[angle_list[key]['parent'], :]
            child = keypoints[angle_list[key]['child'], :]
            u_t = torch.Tensor(angle_list[key]['u']).to(device)
            v_t = torch.Tensor(angle_list[key]['v']).to(device)
            z_t = torch.Tensor(angle_list[key]['z']).to(device)
            if not is_positive:
                u_t[0] = -u_t[0]
                v_t[0] = -v_t[0]

            # calculate the axies of parents
            u = joint - parent
            u_norm = torch.linalg.norm(u)
            u = u / u_norm
            # print("u_norm:", u_norm)
            R = rodriguez_torch(u_t, u)
            v = R @ v_t
            v_norm = torch.linalg.norm(v)
            v = v / v_norm
            # print("v_norm:", v_norm)
            z = torch.cross(u, v)
            z_norm = torch.linalg.norm(z)
            z = z / z_norm
            # print("z_norm:", z_norm)

            # calculate the angle in spherical coordinate
            b = child - joint
            b_norm = torch.linalg.norm(b)
            b_uv = b - torch.dot(z, b) * z
            b_uv_norm = torch.linalg.norm(b_uv)
            x_theta = torch.dot(z, b) / (b_norm)
            # print("b_norm:", b_norm)
            # print("b_uv_norm:", b_uv_norm)
            # x_theta = torch.where(x_theta >= -1, x_theta, torch.full(x_theta.shape, -1+(1e-4), device=device).type(torch.float32))
            # x_theta = torch.where(x_theta <= 1, x_theta, torch.full(x_theta.shape, 1-(1e-4), device=device).type(torch.float32))
            x_theta = torch.clamp(x_theta, min=-1 + (1e-4), max=1 - (1e-4))
            # print("x_theta:", x_theta)
            theta = torch.arccos(x_theta)
            # print("theta:", theta)
            if b_uv_norm.item() == 0:
                phi = torch.zeros_like(theta, device=device)
            else:
                x_phi = torch.dot(u, b_uv) / (b_uv_norm)
                # x_phi = torch.where(x_phi >= -1, x_phi, torch.full(x_phi.shape, -1+(1e-4), device=device).type(torch.float32))
                # x_phi = torch.where(x_phi <= 1, x_phi, torch.full(x_phi.shape, 1-(1e-4), device=device).type(torch.float32))
                x_phi = torch.clamp(x_phi, min=-1 + (1e-4), max=1 - (1e-4))
                # print("x_phi:", x_phi)
                phi = torch.arccos(x_phi)
                # print("phi:", phi)
            theta = 180 * (theta / pi)
            phi = 180 * (phi / pi)
            # print("theta:", theta)
            # print("phi:", phi)
            vec_dir = torch.cross(u, b)
            if torch.dot(vec_dir, z).item() < 0:
                phi = 360 - phi
            angle_res[i, 0] = theta
            angle_res[i, 1] = phi
        return angle_res

    def bone_length_mean(self, bl_vector):
        '''
        各肢体骨骼长度统计均值
        input: 
            bl_vector: numpy (B,N,3) batchsize, bone part number, (x,y,z)
        '''
        # print("Start calculate bone length mean, input data shape is:", bl_vector.shape)
        bl_np = np.sqrt(np.sum(np.square(bl_vector), axis=2))
        bl_mean = np.mean(bl_np, axis=0)
        # print("Successfully calculate bone length mean, shape is:", bl_mean.shape)
        return bl_mean

    def bl_eva(self, bl_gt, bl_propor_delta):
        '''
        evaluate the bone length proportion (estimation/gt(or mean))
        input:
            bl_gt: numpy (N, 1) bone number
            bl_propor_delta: proportion tolerance propor_min=1-bl_propor_delta propor_max=1+bl_propor_delta
        '''
        sample_nums = self.keypoints.shape[0]
        bl_vector = self.keypoints2bone()
        bl_length = np.sqrt(np.sum(np.square(bl_vector), axis=2))
        bl_propor = bl_length / bl_gt
        bl_eva = np.ones((sample_nums, 1))
        bl_propor_min = np.full(16, 1 - bl_propor_delta)
        bl_propor_max = np.full(16, 1 + bl_propor_delta)
        for num in range(sample_nums):
            if (bl_propor[num] > bl_propor_min).all() and (
                    bl_propor[num] < bl_propor_max).all():
                pass
            else:
                # print(bl_propor[num])
                bl_eva[num] = 0

        bl_eva_bone = np.where(bl_propor < bl_propor_min, 0, bl_propor)
        bl_eva_bone = np.where(bl_eva_bone > bl_propor_max, 0, bl_eva_bone)
        bl_eva_bone = np.where(bl_eva_bone != 0, 1, 0)
        # print("bl_bone:", bl_eva_bone)
        # print(bl_eva_bone.shape)
        bl_eva_bone = np.sum(bl_eva_bone, axis=1) / bl_propor.shape[1]
        # print("bl_bone:", bl_eva_bone)
        # print(bl_eva_bone.shape)
        return bl_eva, bl_eva_bone

    def angle_eva(self, occupany_matrix):
        '''
        evaluate the joint angles
        input:
            occupany_matrix: numpy (N, 181, 361) joint_angle_nums theta:0-180 phi:0-360  occupany_matrix[joint_i, theta, phi]==1 if the angle is reasonable otherwise ==0
        '''
        sample_nums = self.keypoints.shape[0]
        angle_nums = len(self.joint_angle)
        keypoints = self.keypoints
        angle_eva = np.ones((sample_nums, 1))
        angle_eva_joint = np.zeros((sample_nums, 1))
        angle_res = None

        # evaluate joints angles
        for sample_i in range(sample_nums):
            keypoint = keypoints[sample_i]
            angle_res = self.keypoints2angle(keypoint)
            is_real = True
            joint_angle_re_num = 0
            for angle_i, key in enumerate(self.joint_angle):
                o_matrix = occupany_matrix[angle_i]
                angle = angle_res[angle_i]
                if (o_matrix[int(angle[0]), int(angle[1])] == 0):
                    # print("{}'s angle is unreasonable.".format(self.joint_angle[key]['name']))
                    is_real = False
                else:
                    joint_angle_re_num += 1
            angle_eva_joint[sample_i] = joint_angle_re_num / angle_nums
            if is_real is not True:
                angle_eva[sample_i] = 0

        return angle_eva, angle_eva_joint

    def angle_gmm_score(self):
        '''[calculate joint angles gmm score]

        Returns:
            [numpy (N, K, 1)]: [gmm score of joint angles and change the range: [0, gmm_score_3_sigma]-> [5, -5]]
        '''
        sample_nums = self.keypoints.shape[0]
        angle_nums = len(self.joint_angle)
        keypoints = self.keypoints
        angle_list = self.joint_angle
        angle_scores = np.ones((sample_nums, angle_nums, 1))
        angle_res = np.zeros((sample_nums, angle_nums, 2))

        # score joints angles
        for sample_i in range(sample_nums):
            keypoint = keypoints[sample_i]
            angle_res[sample_i] = self.keypoints2angle(keypoint)

        angle_sin = np.zeros((sample_nums, angle_nums, 3))
        angle_sin[:, :, 0] = np.sin(angle_res[:, :, 0])
        angle_sin[:, :, 1] = np.sin(angle_res[:, :, 1])
        angle_sin[:, :, 2] = np.cos(angle_res[:, :, 1])

        for i, key in enumerate(angle_list):
            save_path = os.path.join(self.save_folder,
                                     angle_list[key]['name'] + "_gmm.pkl")
            gmm_score = angle_list[key]['gmm_score']
            gmm = joblib.load(save_path)
            scores = np.exp(gmm.score_samples(angle_sin[:, i, :]))
            angle_scores[:, i, :] = (scores -
                                     gmm_score / 2) * (2 / gmm_score) * (-5)

        return angle_scores

    def angle_gmm_score_torch(self):
        '''[calculate joint angles gmm score]

        Returns:
            [torch Tensor (N, K, 1)]: [gmm score of joint angles and change the range: [0, gmm_score_3_sigma]-> [5, -5]]
        '''
        sample_nums = self.keypoints.shape[0]
        angle_nums = len(self.joint_angle)
        keypoints = self.keypoints
        device = keypoints.device
        angle_list = self.joint_angle
        angle_scores = torch.zeros(sample_nums, angle_nums, 1, device=device)
        angle_res = torch.zeros(sample_nums, angle_nums, 2, device=device)

        # score joints angles
        for sample_i in range(sample_nums):
            keypoint = keypoints[sample_i]
            angle_res[sample_i] = self.keypoints2angle_torch(keypoint)
        # print("angle_res:", angle_res)
        angle_sin = torch.zeros(sample_nums, angle_nums, 3, device=device)
        angle_sin[:, :, 0] = torch.sin(angle_res[:, :, 0])
        angle_sin[:, :, 1] = torch.sin(angle_res[:, :, 1])
        angle_sin[:, :, 2] = torch.cos(angle_res[:, :, 1])

        for i, key in enumerate(angle_list):
            save_path = os.path.join(self.save_folder,
                                     angle_list[key]['name'] + "_gmm.pkl")
            gmm_score = angle_list[key]['gmm_score']
            gmm_model = joblib.load(save_path)
            means = torch.FloatTensor(gmm_model.means_).to(device)
            convariances = torch.FloatTensor(gmm_model.covariances_).to(device)
            angle_sin_numpy = angle_sin[:, i, :].detach().cpu().numpy()
            probs = gmm_model.predict_proba(angle_sin_numpy)
            component_size = probs.shape[1]
            for component_i in range(component_size):
                prob_i = torch.FloatTensor(probs[:, component_i]).to(device)
                mean = means[component_i]
                convariance = convariances[component_i]
                for sample_i in range(sample_nums):
                    score = gmm.gmm_scores_cal_torch(
                        angle_sin[sample_i, i, :].unsqueeze(0), mean,
                        convariance)
                    angle_scores[sample_i,
                                 i, :] += (prob_i[sample_i] * score).squeeze(0)
            # print("angle_scores:", angle_scores[:, i, :])
            # print("gmm_score:", gmm_score)
            angle_scores[:, i, :] = (angle_scores[:, i, :] -
                                     gmm_score / 2) * (2 / gmm_score) * (-5)
        # print("angle_scores:", angle_scores)
        return angle_scores