"""
Author: jenningsliu
Date: 2022-06-24 16:57:04
LastEditors: jenningsliu
LastEditTime: 2022-07-07 13:41:49
FilePath: /nerf-loc/datasets/video/covisibility_sampler.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import numpy as np
import cv2
import os
from collections import defaultdict

# from models.ops.pointnet2.pointnet2_batch import pointnet2_utils
from .furthest_pose_sampler import get_next_FPS_sample

class CovisibilitySampling(object):
    def __init__(self, dataset, max_num_pts=8192):
        self.dataset = dataset
        if len(pts3d) < 8192:
            # pts3d = torch.tensor(dataset.pc.vertices).float().cuda()
            # rand_idx = pointnet2_utils.furthest_point_sample(pts3d.view(1,-1,3), 8192)[0].long()
            # self.pc = pts3d[rand_idx].cpu().numpy()
            pts3d = dataset.pc.vertices
            rand_idx = np.random.choice(len(pts3d), 8192, replace=False)
            self.pc = pts3d[rand_idx]
        self.ref_poses = dataset.ref_poses
        self.ref_Ks = dataset.ref_intrinsics

        self.points_to_images = {}
        self.images_to_points = {}

        self.build_visibility()
    
    def get_visible_points(self, T, K):
        xyz_cam = T[:3,:3] @ self.pc.T + T[:3,3:4]
        uvz = K @ xyz_cam
        u,v,z = uvz[0], uvz[1], uvz[2]
        u = u / (z+1e-8)
        v = v / (z+1e-8)
        w = int(K[0,2] * 2)
        h = int(K[1,2] * 2)
        mask = (z > 0) & (u > 0) & (u < w) & (v > 0) & (v < h)
        return np.nonzero(mask)[0]

    def build_visibility(self):
        for name, Tcw in self.ref_poses.items():
            cam_params = self.ref_Ks[name]
            K = np.eye(3)
            K[0,0] = cam_params[0]
            K[1,1] = cam_params[1]
            K[0,2] = cam_params[2]
            K[1,2] = cam_params[3]
            vis_idx = self.get_visible_points(Tcw, K)
            self.images_to_points[name] = vis_idx

    def find_visible_images(self, pts_idx):
        pass

    def find_visible_points(self, imgs_idx):
        pass

    def sample(self, max_k, target_idx=None):
        # print(f'build core image set for scene: {self.dataset.scene}')
        if target_idx is None:
            target_idx = set(np.arange(len(self.pc)))
        samples = {}
        candidates = {name:self.ref_poses[name] for name in self.images_to_points.keys()}
        for k in range(max_k):
            if len(target_idx) > 0: 
                best = None
                best_overlap = None
                for ref_name in candidates:
                    vis_idx = self.images_to_points[ref_name]
                    intersection = target_idx.intersection(set(vis_idx))
                    if best is None or len(intersection) > len(best_overlap):
                        best = ref_name
                        best_overlap = intersection
                target_idx = target_idx - best_overlap
                samples[best] = self.ref_poses[best]
                candidates.pop(best)
            else:
                # if all points are obversed, pick furthest pose
                next_idx, next_name, next_pose = get_next_FPS_sample(candidates, samples)
                samples[next_name] = next_pose
                candidates.pop(next_name)
            # print(len(target_idx))
        # print(f'Final core image set size of {self.dataset.scene} is {len(samples)}')
        return list(samples.keys())

    def find_covisible_images(self, T, K, max_k):
        target_idx = set(self.get_visible_points(T, K))

        samples = {}
        candidates = {name:self.ref_poses[name] for name in self.images_to_points.keys()}
        for k in range(max_k):
            if len(target_idx) == 0:
                break
            best = None
            best_overlap = None
            for ref_name in candidates:
                vis_idx = self.images_to_points[ref_name]
                intersection = target_idx.intersection(set(vis_idx))
                if best is None or len(intersection) > len(best_overlap):
                    best = ref_name
                    best_overlap = intersection
            target_idx = target_idx - best_overlap
            samples[best] = self.ref_poses[best]
            candidates.pop(best)
        return list(samples.keys())

if __name__ == '__main__':
    from configs import config_parser
    from datasets import build_dataset
    from utils.common import colorize_np
    parser = config_parser()
    args = parser.parse_args()

    multi_trainset = build_dataset(args, 'train')
    # multi_testset = build_dataset(args, 'test')

    trainset = multi_trainset.datasets[0]
    # testset = multi_testset.datasets[0]
    cov = CovisibilitySampling(trainset)

    # target_idx = np.random.choice(len(testset), 1)[0]
    # print(target_idx)
    # target_frame = testset[target_idx]
    # target_name = target_frame['filename']
    # target_pose = target_frame['pose']
    # target_K = target_frame['K']
    # res = cov.find_covisible_images(target_pose, target_K, max_k=5)
    # images = [cv2.imread(os.path.join(testset.root_dir, target_name))]

    res = cov.sample(max_k=args.image_core_set_size)
    print('train: ', res)
    # cov1 = CovisibilitySampling(testset)
    # res1 = cov1.sample(max_k=args.image_core_set_size)
    # print('test: ', res1)
    images = []
    depths = []

    # vis
    for p in res:
        idx = trainset.ref_image_idx[p]
        meta = trainset.train_meta_info_list[idx]
        filename = os.path.join(trainset.root_dir, p)
        images.append(cv2.imread(filename))
        depth_filename = meta['depth_file_name']
        depths.append(cv2.imread(os.path.join(trainset.root_dir, depth_filename), cv2.IMREAD_ANYDEPTH) / 1000)
    color = np.concatenate(images, axis=1)
    depth = np.concatenate(depths, axis=1)
    depth = (colorize_np(depth, 'jet', None, range=None, append_cbar=False, cbar_in_image=False) * 255).astype(np.uint8)
    vis = np.concatenate([color, depth], axis=0)
    vis = cv2.resize(vis, fx=0.25, fy=0.25, dsize=None)
    cv2.imwrite('vis_coreset.png', vis)
