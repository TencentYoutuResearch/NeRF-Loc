"""
Author: jenningsliu
Date: 2022-03-10 17:46:34
LastEditors: jenningsliu
LastEditTime: 2022-03-29 19:57:41
FilePath: /nerf-loc/datasets/video/preprocess_12scenes.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import sys
import glob
import pickle as pkl
import os
from os import path as osp
import numpy as np
import trimesh
import cv2
from tqdm import tqdm

def load_pose(pose_txt):
    pose = []
    with open(pose_txt, 'r') as f:
        for line in f:
            row = line.strip('\n').split()
            row = [float(c) for c in row]
            pose.append(row)
    pose = np.array(pose).astype(np.float32)
    assert pose.shape == (4,4)
    return pose

def build_meta_infos(data_folder, place, scene, images, poses, color_focal, color_width, color_height):
    meta_infos = []
    for image, pose in zip(images, poses):
        # some image have invalid pose files, skip those
        valid = True
        with open(ds + '/' + scene + '/data/' + pose, 'r') as f:
            pose_file = f.readlines()
            for line in pose_file:
                if 'INF' in line:
                    valid = False
        if not valid:
            continue

        Twc = load_pose(os.path.join(data_folder, pose))
        depth = cv2.imread(
            os.path.join(data_root, place, scene, 'data', image).replace('color.jpg', 'depth.png'), 
            cv2.IMREAD_ANYDEPTH)
        depth = depth.astype(np.float32)/1000
        depth = depth.reshape(-1)
        near = np.percentile(depth, 0.1)
        far = np.percentile(depth, 99.9)
        # link_frame(i, 'test')
        meta_infos.append({
            'file_name': os.path.join(place, scene, 'data', image),
            'frame_id': int(image.split('.')[0].split('-')[1]),
            'sequence_id': '0',
            'depth_file_name': os.path.join(place, scene, 'data', image).replace('color.jpg', 'depth.png'),
            'extrinsic_Tcw': np.linalg.inv(Twc)[:3],
            'camera_intrinsic': np.array([
                color_focal, color_focal, color_width/2, color_height/2,   0.,   0.
            ], dtype=np.float32),
            'frame_dim': (color_height, color_width),
            'near': near,
            'far': far
        })
    return meta_infos

if __name__ == '__main__':
    data_root = sys.argv[1]
    for place in ['apt1', 'apt2', 'office1', 'office2']:
    # for place in ['apt1']:
        ds = os.path.join(data_root, place)
        scenes = os.listdir(ds)

        for scene in scenes:

            data_folder = ds + '/' + scene + '/data/'

            if not os.path.isdir(data_folder):
                # skip README files
                continue

            print(f"Processing files for {ds}/{scene}")

            # read the train / test split - the first sequence is used for testing, everything else for training
            with open(ds + '/' + scene + '/split.txt', 'r') as f:
                split = f.readlines()
            split = int(split[0].split()[1][8:-1])

            # read the calibration parameters, we use only the color_focal
            with open(ds + '/' + scene + '/info.txt', 'r') as f:
                info_lines = f.readlines()
            color_focal = info_lines[7].split()
            color_focal = (float(color_focal[2]) + float(color_focal[7])) / 2
            color_width = int(info_lines[2].split()[-1])
            color_height = int(info_lines[3].split()[-1])

            files = os.listdir(data_folder)

            images = [f for f in files if f.endswith('color.jpg')]
            images.sort()

            poses = [f for f in files if f.endswith('pose.txt')]
            poses.sort()

            # frames up to split are test images
            test_meta_infos = build_meta_infos(data_folder, place, scene, 
                images[:split], poses[:split], color_focal, color_width, color_height)
            train_meta_infos = build_meta_infos(data_folder, place, scene, 
                images[split:], poses[split:], color_focal, color_width, color_height)

            print('near: ', np.array([m['near'] for m in train_meta_infos]).min())
            print('far: ', np.array([m['far'] for m in train_meta_infos]).max())
            with open(osp.join(ds, scene, f'info_train.pkl'), 'wb') as fout:
                pkl.dump(train_meta_infos, fout)
            with open(osp.join(ds, scene, f'info_test.pkl'), 'wb') as fout:
                pkl.dump(test_meta_infos, fout)
            print(f'test_num: {len(test_meta_infos)} train_num: {len(train_meta_infos)}, total: {len(images)}')

            model_path = glob.glob(ds + '/' + scene + '/*.ply')[0]
            mesh = trimesh.load(model_path)
            cloud = trimesh.PointCloud(vertices=mesh.vertices)
            cloud.export(ds + '/' + scene + '/pc.ply')
            

        