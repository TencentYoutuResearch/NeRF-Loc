"""
Author: jenningsliu
Date: 2022-03-29 22:36:11
LastEditors: jenningsliu
LastEditTime: 2022-08-22 13:04:44
FilePath: /nerf-loc/datasets/llff.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
from torch.utils.data import Dataset
import torchvision
import numpy as np
import random

import sys
sys.path.append('third_party/NeuRay')
from third_party.NeuRay.dataset.train_dataset import *

class NeurayBaseDataset(Dataset):
    def __init__(self, args, split, database_name):
        super().__init__()
        self.args = args
        self.database = parse_database_name(database_name)
        self.ref_ids, test_que_ids = get_database_split(self.database, 'test')
        self.split = split
        if split == 'train':
            self.que_ids = self.ref_ids
        else:
            self.que_ids = test_que_ids
        self.scale_factor = 1

    def set_mode(self, mode):
        pass

    def prepare_pose(self, poses):
        hom_poses = np.tile(np.eye(4, dtype=np.float32)[None], [len(poses),1,1]).copy()
        hom_poses[:,:3] = poses
        return np.linalg.inv(hom_poses)

    def select_working_views(self, database, que_id, ref_ids):
        ref_ids = [i for i in ref_ids if i != que_id]
        database_name = database.database_name
        dist_idx = compute_nearest_camera_indices(database, [que_id], ref_ids)[0]
        dist_idx = dist_idx[:10]
        ref_ids = np.array(ref_ids)[dist_idx]
        return ref_ids

    def __len__(self):
        # return len(self.images)
        return len(self.que_ids)

    def __getitem__(self, idx):
        que_id = self.que_ids[idx]
        ref_ids = self.select_working_views(self.database, que_id, self.ref_ids)
        ref_imgs_info = build_imgs_info(self.database, ref_ids, -1, True)
        que_imgs_info = build_imgs_info(self.database, [que_id], has_depth=self.split=='train')
        ref_imgs_info = pad_imgs_info(ref_imgs_info, 16)
        near, far = que_imgs_info['depth_range'][0]
        ret = {
            'scene': self.args.scene,
            'filename': que_imgs_info['img_ids'][0]+'.png',
            'image': que_imgs_info['imgs'][0],
            'pose': self.prepare_pose(que_imgs_info['poses'])[0],
            'K': que_imgs_info['Ks'][0],
            'near': near,
            'far': far,
            'topk_images': ref_imgs_info['imgs'],
            'topk_poses': self.prepare_pose(ref_imgs_info['poses']),
            'topk_Ks': ref_imgs_info['Ks'],
            'topk_depths': ref_imgs_info['depth'].squeeze(1),
            'points3d': np.zeros([8,3],dtype=np.float32),
            'scale_factor': self.scale_factor
        }
        if 'depth' in que_imgs_info:
            ret['depth'] = que_imgs_info['depth'].squeeze(1)[0]
        else:
            ret['depth'] = np.zeros_like(ret['image'][0])
        if 'true_depth' in ref_imgs_info:
            ret['topk_depths_gt'] = ref_imgs_info['true_depth'].squeeze(1)
        return ret

if __name__ == '__main__':
    from configs import config_parser
    parser = config_parser()
    args = parser.parse_args()
    dataset = LLFFDataset(args, 'train')
    # print(dataset[0])
    for k,v in dataset[0].items():
        if type(v) != str:
            print(k, v.shape)
    # print(len(dataset))
