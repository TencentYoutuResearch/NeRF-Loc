"""
Author: jenningsliu
Date: 2022-03-10 17:46:34
LastEditors: jenningsliu
LastEditTime: 2022-08-07 17:55:57
FilePath: /nerf-loc/datasets/video/preprocess_7scenes.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import sys
import glob
import pickle as pkl
import os
from os import path as osp
import numpy as np
import cv2
from tqdm import tqdm
import re

import fusion

data_root = sys.argv[1]

focallength = 525.0
# focallength = 585.0

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

def fuse_tsdf(data_path, seqs):
    cam_intr = np.array([
        [focallength, 0, 320.0],
        [0,focallength,240.0],
        [0,0,1]
    ])
    vol_bnds = np.zeros((3,2))
    for seq in seqs:
        seq = int(seq.replace('sequence', ''))
        seq_folder = osp.join(data_path, 'seq-%02d'%seq)
        for img in tqdm(glob.glob(seq_folder+'/*color.png')):
            mat = re.search(r'frame-(\d+)', img)
            i = int(mat.group(1))
            # Read depth image and camera pose
            depth_im = cv2.imread(
                f"{data_path}/rendered_depth/train/depth/seq%02d_frame-%06d.pose.depth.tiff"%(seq, i),-1
            ).astype(float)
            depth_im /= 1000.  # depth is saved in 16-bit PNG in millimeters
            depth_im[depth_im == 65.535] = 0  # set invalid depth to 0 (specific to 7-scenes dataset)
            cam_pose = \
                np.loadtxt(f"{data_path}/seq-%02d/frame-%06d.pose.txt"%(seq, i))  # 4x4 rigid transformation matrix

            # Compute camera view frustum and extend convex hull
            view_frust_pts = fusion.get_view_frustum(depth_im, cam_intr, cam_pose)
            vol_bnds[:,0] = np.minimum(vol_bnds[:,0], np.amin(view_frust_pts, axis=1))
            vol_bnds[:,1] = np.maximum(vol_bnds[:,1], np.amax(view_frust_pts, axis=1))
    print(vol_bnds)

    # Initialize voxel volume
    print("Initializing voxel volume...")
    tsdf_vol = fusion.TSDFVolume(vol_bnds, voxel_size=0.02)

    # Loop through RGB-D images and fuse them together
    for seq in seqs:
        seq = int(seq.replace('sequence', ''))
        seq_folder = osp.join(data_path, 'seq-%02d'%seq)
        for img in tqdm(glob.glob(seq_folder+'/*color.png')):
            mat = re.search(r'frame-(\d+)', img)
            i = int(mat.group(1))
            if i % 5 != 0:
                continue

            # Read RGB-D image and camera pose
            color_image = \
                cv2.cvtColor(cv2.imread(f"{data_path}/seq-%02d/frame-%06d.color.png"%(seq,i)), cv2.COLOR_BGR2RGB)
            #depth_im = cv2.imread(f"{data_path}/frame-%06d.depth.png"%(i),-1).astype(float)
            depth_im = cv2.imread(
                f"{data_path}/rendered_depth/train/depth/seq%02d_frame-%06d.pose.depth.tiff"%(seq, i),-1
            ).astype(float)
            depth_im /= 1000.
            depth_im[depth_im == 65.535] = 0
            cam_pose = np.loadtxt(f"{data_path}/seq-%02d/frame-%06d.pose.txt"%(seq,i))

            # Integrate observation into voxel volume (assume color aligned with depth)
            tsdf_vol.integrate(color_image, depth_im, cam_intr, cam_pose, obs_weight=1.)

    # Get mesh from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving mesh to mesh.ply...", osp.join(data_path, "mesh.ply"))
    verts, faces, norms, colors = tsdf_vol.get_mesh()
    fusion.meshwrite(osp.join(data_path, "mesh.ply"), verts, faces, norms, colors)

    # Get point cloud from voxel volume and save to disk (can be viewed with Meshlab)
    print("Saving point cloud to pc.ply...", osp.join(data_path, "pc.ply"))
    point_cloud = tsdf_vol.get_point_cloud()
    fusion.pcwrite(osp.join(data_path, "pc.ply"), point_cloud)


def process_split(scene_folder, seqs, split):
    if split == 'train':
        fuse_tsdf(scene_folder, seqs)
    meta_infos = []
    for seq in seqs:
        num = int(seq.replace('sequence', ''))
        seq_folder = osp.join(scene_folder, 'seq-%02d'%num)
        for img in tqdm(glob.glob(seq_folder+'/*color.png')):
            img_name = img.replace(data_root, '').lstrip('/')
            Twc = load_pose(img.replace('color.png', 'pose.txt'))
            image = cv2.imread(os.path.join(data_root, img_name))

            mat = re.search(r'frame-(\d+)', img)
            i = int(mat.group(1))
            render_depth_file = \
                f"{scene_folder}/rendered_depth/train/depth/seq%02d_frame-%06d.pose.depth.tiff"%(num, i)
            if split == 'train':
                depth = cv2.imread(render_depth_file,-1)
            else:
                # only use to compute far near
                depth = cv2.imread(os.path.join(data_root, img_name.replace('color', 'depth')), cv2.IMREAD_ANYDEPTH)

            depth[depth==65535] = 0
            depth = depth.astype(np.float32)/1000
            depth = depth.reshape(-1)
            near = np.percentile(depth, 0.1)
            far = np.percentile(depth, 99.9)
            meta_infos.append({
                'file_name': img_name,
                'frame_id': i,
                'sequence_id': num,
                'depth_file_name': render_depth_file.replace(data_root, '').lstrip('/'),
                'extrinsic_Tcw': np.linalg.inv(Twc)[:3],
                'camera_intrinsic': np.array([focallength, focallength, 320., 240.,   0.,   0.], dtype=np.float32),
                'frame_dim': image.shape[:2],
                'near': near,
                'far': far
            })
    print(meta_infos[0])
    with open(osp.join(scene_folder, f'info_{split}.pkl'), 'wb') as fout:
        pkl.dump(meta_infos, fout)
    return meta_infos

for scene in ['chess', 'pumpkin', 'fire', 'heads', 'office', 'redkitchen', 'stairs']:
    scene_folder = osp.join(data_root, scene)

    train_split = []
    with open(osp.join(data_root, scene, 'TrainSplit.txt'), 'r') as f:
        for line in f:
            train_split.append(line.strip('\n'))
    process_split(scene_folder, train_split, 'train')

    test_split = []
    with open(osp.join(data_root, scene, 'TestSplit.txt'), 'r') as f:
        for line in f:
            test_split.append(line.strip('\n'))
    process_split(scene_folder, test_split, 'test')


    