"""
Author: jenningsliu
Date: 2022-08-03 12:05:07
LastEditors: jenningsliu
LastEditTime: 2022-08-23 15:38:53
FilePath: /nerf-loc/datasets/colmap_dataset.py
Description: 
    Dataloader for Colmap format data
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import re
from collections import defaultdict
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision
import pickle as pkl
from skimage.io import imread
import os.path as osp
import cv2
import os
import trimesh
import copy
import numpy as np
from utils.common import batched_angular_dist_rot_matrix

from datasets.colmap.read_write_model import read_model,qvec2rotmat
from datasets.video.transform import ResizeAndCrop
from utils.visualization import project_3d_points
from datasets.video.reader import read_array, read_pfm
from datasets.furthest_pose_sampler import FurtherPoseSampling

class ColmapDataset(Dataset):
    def __init__(self, args, dense_path, split, depth_type='mvsnet'):
        """
            args: config
            dense_path: estimated dense depth map folder
            split: train/test
            depth_type: algorithm used to compute depth map, colmap/mvsnet
        """
        sparse_path = os.path.join(dense_path, 'sparse')
        image_path = os.path.join(dense_path, 'images')
        self.dense_path = dense_path
        self.image_path = image_path
        self.depth_type = depth_type
        self.cameras, self.images, self.points3D = read_model(sparse_path)
        self.image_ids = [img_id for img_id,img in self.images.items()]
        self.image_ids = sorted(self.image_ids, key=lambda x:self.images[x].name)
        train_image_ids = self.get_split_image_ids('train')
        test_image_ids = self.get_split_image_ids('test')
        self.ref_image_ids = train_image_ids
        self.current_image_ids = train_image_ids if split == 'train' else test_image_ids
        # self.image_ids = self.image_ids[200:300]
        self.near, self.far = self.compute_near_far()
        self.transform = ResizeAndCrop(target_size=256, base_image_size=16)
        self.scale_factor = 1
        core_image_ids = self.sample_coreset(args.image_core_set_size)
        self.image_core_set = self.load_support_images(core_image_ids)

    def get_split_image_ids(self, split):
        split_seqs = []
        with open(os.path.join(self.dense_path, f'{split}.txt'), 'r') as f:
            for line in f:
                split_seqs.append(line.strip('\n'))
        split_image_ids = []
        for img_id in self.image_ids:
            if self.images[img_id].name.split('/')[-2] in split_seqs:
                split_image_ids.append(img_id)
        return split_image_ids

    def load_support_images(self, topk_image_ids):
        topk_images, topk_depths, topk_poses, topk_Ks = [],[],[],[]
        for image_id in topk_image_ids:
            rgb, depth, w2c, K, mask = self.load_frame(image_id)
            topk_images.append(rgb.transpose(2,0,1).astype(np.float32) / 255.)
            topk_depths.append(depth)
            topk_poses.append(np.linalg.inv(w2c).astype(np.float32))
            topk_Ks.append(K.astype(np.float32))
        topk_images = np.array(topk_images)
        topk_depths = np.array(topk_depths)
        topk_poses = np.array(topk_poses)
        topk_Ks = np.array(topk_Ks)
        return topk_images, topk_depths, topk_poses, topk_Ks
    
    def sample_coreset(self, max_k):
        self.ref_poses = {
            img.id:self.parse_colmap_pose(img) for img_id,img in self.images.items() if img_id in self.ref_image_ids
        }
        return FurtherPoseSampling.sample_FPS(self.ref_poses, max_k)

    def set_mode(self, mode):
        self.mode = mode

    def compute_near_far(self):
        xyz = np.array([p.xyz for p in self.points3D.values()])
        nears = []
        fars = []
        for img_id in self.image_ids:
            img = self.images[img_id]
            w2c = self.parse_colmap_pose(img)
            camera = self.cameras[img.camera_id]
            K = self.get_intrinsic_matrix(camera)
            uv, z = project_3d_points(xyz,w2c,K)
            u,v = uv[:,0], uv[:,1]
            H,W = camera.height, camera.width
            mask = (u >= 0) & (v >= 0) & (u < W) & (v < H) & (z > 0)
            near = np.percentile(z[mask], 0.1)
            far = np.percentile(z[mask], 99.)
            nears.append(near)
            fars.append(far)
        near = np.array(nears).min()
        far = np.array(fars).max()
        return near, far

    def get_intrinsic_matrix(self, camera):
        assert camera.model == 'PINHOLE'
        fx,fy,cx,cy = camera.params
        K = np.array([
            [fx, 0, cx],
            [0,fy,cy],
            [0,0,1]
        ])
        return K

    def parse_colmap_pose(self, image):
        R = qvec2rotmat(image.qvec)
        t = image.tvec
        Tcw = np.eye(4)
        Tcw[:3,:3] = R
        Tcw[:3,3] = t
        return Tcw

    def read_image(self, image):
        img = cv2.imread(os.path.join(self.image_path, image.name))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        H,W = img.shape[:2]
        return img

    def load_colmap_depth(self, img_name, W, H):
        img_id = '/'.join(img_name.split('/')[-2:])
        fn = f'{self.dense_path}/stereo/depth_maps/{img_id}.geometric.bin'
        if os.path.exists(fn):
            depth = read_array(fn)
            depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
            return depth
        else:
            return np.zeros([H,W], dtype=np.float32)

    def load_mvsnet_depth(self, image_id, W, H):
        id_mapping = {}
        for i, img_id in enumerate(sorted(self.images.keys())):
            id_mapping[img_id] = i
        fn = f'{self.dense_path}/casmvsnet/depth_est/{id_mapping[image_id]:08}.pfm'
        mask_fn = f'{self.dense_path}/casmvsnet/mask/{id_mapping[image_id]:08}_final.png'
        if os.path.exists(fn):
            mask = cv2.imread(mask_fn, cv2.IMREAD_ANYDEPTH) > 0
            depth = read_pfm(fn)[0] * mask.astype(np.float32)
            depth = cv2.resize(depth, (W,H), interpolation=cv2.INTER_NEAREST)
            return depth
        else:
            return np.zeros([H,W], dtype=np.float32)

    def load_frame(self, image_id):
        image = self.images[image_id]
        camera_id = image.camera_id
        camera = self.cameras[camera_id]
        K = self.get_intrinsic_matrix(camera)
        w2c = self.parse_colmap_pose(image)
        rgb = self.read_image(image) # H,W,3
        if self.depth_type == 'colmap':
            depth = self.load_colmap_depth(image.name, camera.width, camera.height)
        elif self.depth_type == 'mvsnet':
            depth = self.load_mvsnet_depth(image.id, camera.width, camera.height)
        else:
            raise NotImplementedError
        mask = np.ones_like(depth)
        # H,W = rgb.shape[:2]
        if self.transform is not None:
            rgb, depth, w2c, K, mask = self.transform(rgb, depth, w2c, K, mask=mask)
        return rgb, depth, w2c, K, mask

    def __len__(self):
        return len(self.current_image_ids)

    def __getitem__(self, idx):
        image_id = self.current_image_ids[idx]
        image = self.images[image_id]
        rgb, depth, w2c, K, mask = self.load_frame(image_id)

        topk_images, topk_depths, topk_poses, topk_Ks = copy.deepcopy(self.image_core_set)
        data = {
            'scene': '0',
            'filename': image.name,
            # "depth_filename": f'{self.dense_path}/casmvs_depth_maps/{image_id}.tiff', # for generate mvs depth
            'image': rgb.transpose(2,0,1).astype(np.float32) / 255.,
            'pose': np.linalg.inv(w2c).astype(np.float32),
            'K': K.astype(np.float32),
            'near': float(self.near),
            'far': float(self.far),
            'depth': depth.astype(np.float32),
            # 'target_mask': mask.astype(np.bool),
            "topk_poses": topk_poses, # poses of top-k similar images
            'topk_images': topk_images,
            'topk_depths': topk_depths,
            'topk_Ks': topk_Ks,
            'points3d': np.array([p.xyz for p in self.points3D.values()]).astype(np.float32),
            'scale_factor': self.scale_factor
        }
        return data
