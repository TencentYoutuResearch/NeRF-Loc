"""
Author: jenningsliu
Date: 2022-06-28 18:08:20
LastEditors: jenningsliu
LastEditTime: 2022-07-28 19:45:24
FilePath: /nerf-loc/datasets/video/furthest_pose_sampler.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import numpy as np
import cv2
import os
from collections import defaultdict
import copy

from utils.common import batched_angular_dist_rot_matrix

def get_next_FPS_sample(candidates, samples):
    candidates_name = list(candidates.keys())
    candidates_pose = np.array([candidates[n] for n in candidates_name]) # N,4,4
    sampled_name = list(samples.keys())
    sampled_pose = np.array([samples[n] for n in sampled_name]) # M,4,4

    N, M = len(candidates_pose), len(sampled_pose)

    candidates_R = np.tile(candidates_pose[:,None,:3,:3], [1,M,1,1]).reshape(-1,3,3) # N*M,3,3
    sampled_R = np.tile(sampled_pose[None,:,:3,:3], [N,1,1,1]).reshape(-1,3,3) # N*M,3,3
    angular_dists = batched_angular_dist_rot_matrix(candidates_R, sampled_R).reshape(N,M)

    max_min_idx = angular_dists.min(axis=1).argmax(axis=0) # N -> 1
    return max_min_idx, candidates_name[max_min_idx], candidates_pose[max_min_idx]

class FurtherPoseSampling(object):
    def __init__(self, dataset):
        self.dataset = dataset

        self.ref_poses = dataset.ref_poses
        self.filter_ref_poses_without_depth() # only sample reference frame with depth

    def filter_ref_poses_without_depth(self):
        meta_info = {d['file_name']:d for d in self.dataset.train_meta_info_list}
        def is_depth_exist(meta):
            return os.path.exists(os.path.join(meta['base_dir'], meta["depth_file_name"]))
        self.ref_poses = {name:pose for name,pose in self.ref_poses.items() if is_depth_exist(meta_info[name])}

    def sample(self, max_k):
        return FurtherPoseSampling.sample_FPS(self.ref_poses, max_k)
    
    @staticmethod
    def sample_FPS(ref_poses, max_k):
        samples = {}
        # np.random.seed(666)
        init_idx = np.random.choice(len(ref_poses), 1, replace=False)[0]
        init_name = list(ref_poses.keys())[init_idx]
        samples[init_name] = ref_poses[init_name]
        candidates = copy.deepcopy(ref_poses)
        candidates.pop(init_name)
        for k in range(1, max_k):
            next_idx, next_name, next_pose = get_next_FPS_sample(candidates, samples)
            samples[next_name] = next_pose

        return list(samples.keys())

if __name__ == '__main__':
    from configs import config_parser
    from datasets import build_dataset
    from utils.common import colorize_np
    import trimesh
    parser = config_parser()
    args = parser.parse_args()

    multi_trainset = build_dataset(args, 'train')
    # multi_testset = build_dataset(args, 'test')

    trainset = multi_trainset.datasets[0]
    # testset = multi_testset.datasets[0]
    sampler = FurtherPoseSampling(trainset)

    res = sampler.sample(max_k=args.image_core_set_size)
    print('train: ', res)
    # cov1 = CovisibilitySampling(testset)
    # res1 = cov1.sample(max_k=args.image_core_set_size)
    # print('test: ', res1)
    images = []
    depths = []
    points = []
    point_colors = []
    # vis
    for p in res:
        idx = trainset.ref_image_idx[p]
        data = trainset[idx]
        image = (data['image'].transpose(1,2,0)*255).astype(np.uint8)
        depth = data['depth']
        K = torch.tensor(data['K'])
        c2w = torch.tensor(data['pose'])

        images.append(image)
        depths.append(depth)
        # print(idx, K)
        depth_tensor = torch.tensor(depth)
        v, u = torch.nonzero((depth_tensor > trainset.near) & (depth_tensor < trainset.far), as_tuple=True)
        z = depth_tensor[v,u]
        uv_hom = torch.stack([u,v,torch.ones_like(u)], dim=0).float() # 3,N
        pts3d_cam = torch.matmul(K.inverse(), uv_hom) * z
        pts3d_cam_hom = torch.cat([pts3d_cam, torch.ones_like(pts3d_cam[:1])])
        pts3d_world = torch.matmul(c2w[:3,:3], pts3d_cam) + c2w[:3,3:]
        points.append(pts3d_world.T)
        point_colors.append(torch.tensor(image)[v,u])
    color = cv2.cvtColor(np.concatenate(images, axis=1), cv2.COLOR_BGR2RGB)
    depth = np.concatenate(depths, axis=1)
    depth = (colorize_np(
        depth, 'jet', None, range=[trainset.near, trainset.far], append_cbar=False, cbar_in_image=False
    ) * 255).astype(np.uint8)
    vis = np.concatenate([color, depth], axis=0)
    # vis = cv2.resize(vis, fx=0.25, fy=0.25, dsize=None)
    cv2.imwrite('vis_coreset.png', vis)

    points = torch.cat(points).numpy()
    point_colors = torch.cat(point_colors).numpy()
    cloud = trimesh.PointCloud(
        vertices=points, 
        colors=point_colors)
    cloud.export(f'coreset_pc.ply')
