"""
Author: jenningsliu
Date: 2022-03-10 17:46:34
LastEditors: jenningsliu
LastEditTime: 2022-07-28 19:42:24
FilePath: /nerf-loc/datasets/video/preprocess_onepose.py
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
from pathlib import Path
import shutil

import trimesh
from utils.common import is_inside_box3d

from datasets.colmap.read_write_model import read_model, write_model, Image
from datasets.colmap.cli import run_colmap_mvs

def copy_sfm_model(input_path, output_path):
    cameras, images, points3D = read_model(path=input_path)

    for image_id, image in images.items():
        image_path = '/'.join(images[image_id].name.split('/')[:-3])
        name = images[image_id].name.replace(image_path+'/', '')
        images[image_id] = Image(
            id=image_id, qvec=image.qvec, tvec=image.tvec,
            camera_id=image.camera_id, name=name,
            xys=image.xys, point3D_ids=image.point3D_ids
        )

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_model(cameras, images, points3D, path=output_path, ext='.bin')

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

def load_intrisic(intrisic_txt):
    intrisic = []
    with open(intrisic_txt, 'r') as f:
        for line in f:
            row = line.strip('\n').split()
            row = [float(c) for c in row]
            intrisic.append(row)
    intrisic = np.array(intrisic).astype(np.float32)
    assert intrisic.shape == (3,3)
    return intrisic

def load_box_corners(box_txt):
    corners = []
    with open(box_txt, 'r') as f:
        for line in f:
            row = line.strip('\n').split()
            row = [float(c) for c in row]
            corners.append(row)
    corners = np.array(corners).astype(np.float32).reshape(8,3)
    return corners

def load_projected_box(box_txt):
    corners = []
    with open(box_txt, 'r') as f:
        for line in f:
            row = line.strip('\n').split()
            row = [float(c) for c in row]
            corners.append(row)
    corners = np.array(corners).astype(np.float32).reshape(8,2)
    return corners

def draw_box(img, arranged_points):
    """
    plot arranged_points on img and save to save_path.
    arranged_points is in image coordinate. [[x, y]]
    """
    RADIUS = 10
    COLOR = (255, 255, 255)
    EDGES = [
      [1, 5], [2, 6], [3, 7], [4, 8],  # lines along x-axis
      [1, 3], [5, 7], [2, 4], [6, 8],  # lines along y-axis
      [1, 2], [3, 4], [5, 6], [7, 8]   # lines along z-axis
    ] 
    for i in range(arranged_points.shape[0]):
        x, y = arranged_points[i]
        cv2.circle(
            img,
            (int(x), int(y)), 
            RADIUS,
            COLOR,
            -10
        )
    # for edge in EDGES:
    #     start_points = arranged_points[edge[0]]
    #     start_x = int(start_points[0])
    #     start_y = int(start_points[1])
    #     end_points = arranged_points[edge[1]]
    #     end_x = int(end_points[0])
    #     end_y = int(end_points[1])
    #     cv2.line(img, (start_x, start_y), (end_x, end_y), COLOR, 2)
    return img

def process_scene(scene_path):
    scene_name = osp.basename(scene_path)
    box_corners = load_box_corners(osp.join(scene_path, 'box3d_corners.txt'))

    train_meta_infos = []
    test_meta_infos = []

    seq_names = []
    for seq_folder in glob.glob(scene_path+'/*'):
        if not os.path.isdir(seq_folder):
            continue
        if 'box3d_corners.txt' in seq_folder or 'colmap' in seq_folder:
            continue
        seq_name = osp.basename(seq_folder)
        seq_names.append(seq_name)
    seq_names.sort()
    print(seq_names)

    np.save(Path(scene_path)/'bboxes_3d.npy', box_corners[None])

    # copy from sfm results first
    if os.path.exists(Path(scene_path)/'pc.ply'):
        full_pc = trimesh.load(Path(scene_path)/'pc.ply')
        xyz_world = np.array(full_pc.vertices)
    else:
        raise Exception('copy from sfm results first!')

    in_mask = is_inside_box3d(xyz_world, box_corners[:8])
    xyz_world = xyz_world[in_mask]
    cloud = trimesh.PointCloud(vertices=xyz_world, colors=full_pc.colors[in_mask])
    cloud.export(Path(scene_path)/'in_box_pc.ply')
    # sys.exit(0)

    for seq_folder in glob.glob(scene_path+'/*'):
        if not os.path.isdir(seq_folder):
            continue
        if 'box3d_corners.txt' in seq_folder or 'colmap' in seq_folder:
            continue
        seq_name = osp.basename(seq_folder)
        img_names = os.listdir(osp.join(seq_folder, 'color'))
        img_names.sort(key=lambda x:int(x.split('.')[0]))
        # vidcap = cv2.VideoCapture(osp.join(seq_folder,'Frames.m4v'))
        for img_name in img_names:
            frame_id = img_name.split('.')[0]
            K = load_intrisic(osp.join(seq_folder, f'intrin_ba/{frame_id}.txt'))
            fx,fy,cx,cy = K[0,0], K[1,1], K[0,2], K[1,2]

            Tcw = load_pose(osp.join(seq_folder, f'poses_ba/{frame_id}.txt'))
            R = Tcw[:3,:3]
            t = Tcw[:3,3]

            xyz_cam_i = (R @ xyz_world.T + t.reshape(3,1)).T
            # box_corners_cam = (R @ box_corners.T + t.reshape(3,1)).T
            # ########
            # # proj_corners = load_projected_box(osp.join(seq_folder, f'reproj_box/{frame_id}.txt'))
            # # success,img = vidcap.read()
            # # img = draw_box(img, proj_corners)
            # # cv2.imwrite(f'onepose_vis_full/{seq_name}/{img_name}', img)

            uvz = K @ xyz_cam_i.T
            uv = uvz[:2] / uvz[2:]
            z = xyz_cam_i[:,2]
            H, W = 512, 512
            valid_mask = (uv[0] >= 0) & (uv[1] >= 0) & (uv[0] < W) & (uv[1] < H) & (z > 0)
            if valid_mask.sum() == 0:
                print('Warining: skip bad pose')
                continue
            # img = cv2.imread(os.path.join(scene_path, seq_name, 'color', img_name))
            # img = draw_box(img, uv.T)
            # os.makedirs(f'onepose_vis/{seq_name}', exist_ok=True)
            # cv2.imwrite(f'onepose_vis/{seq_name}/{img_name}', img)
            # # from IPython import embed;embed()
            # ########
            # use box corners depth to determin range
            # z = box_corners_cam[:,2]
            z = z[z>0] # filter out points that lie behind the cam
            near = z.min()
            far = z.max()

            info = {
                'file_name': os.path.join(scene_name, seq_name, 'color', img_name),
                'frame_id': int(frame_id),
                'sequence_id': seq_name,
                'depth_file_name': os.path.join(scene_name, seq_name, 'depth', img_name.replace('.png', '_mvs.tiff')),
                'extrinsic_Tcw': Tcw[:3],
                'camera_intrinsic': np.array([
                    fx, fy, cx, cy,   0.,   0.
                ], dtype=np.float32),
                'frame_dim': (512, 512),
                'near': near,
                'far': far
            }
            # last sequence as test
            if seq_name == seq_names[-1]:
                test_meta_infos.append(info)
            # elif seq_name == seq_names[0]: # only keep one sequence for train
            elif os.path.exists(os.path.join(os.path.dirname(scene_path), info['depth_file_name'])):
                # skip train image without depth
                train_meta_infos.append(info)
    # print(info)
    meta_infos = train_meta_infos + test_meta_infos
    near = np.array([m['near'] for m in meta_infos]).min()
    far = np.array([m['far'] for m in meta_infos]).max()
    print('near: ', near)
    print('far: ', far)
    print(f'train size: {len(train_meta_infos)} test size: {len(test_meta_infos)}')

    # # split
    # img_ids_train = [id_ for i, id_ in enumerate(img_ids) 
    #                         if files.loc[i, 'split']=='train']
    # img_ids_test = [id_ for i, id_ in enumerate(img_ids)
    #                             if files.loc[i, 'split']=='test']
    # train_meta_infos = [meta_infos[i] for i in img_ids_train]
    # test_meta_infos = [meta_infos[i] for i in img_ids_test]
    # print(train_meta_infos[0])
    # print(f'{len(train_meta_infos)} train, {len(test_meta_infos)} test')
    with open(osp.join(scene_path, f'info_train.pkl'), 'wb') as fout:
        pkl.dump(train_meta_infos, fout)

    with open(osp.join(scene_path, f'info_test.pkl'), 'wb') as fout:
        pkl.dump(test_meta_infos, fout)
    
    # cloud = trimesh.PointCloud(vertices=xyz_world, colors=rgb)
    # cloud.export(os.path.join(scene_path, '../pc.ply'))


if __name__ == '__main__':
    data_root = sys.argv[1]
    sfm_data_root = sys.argv[2] # sfm workspace of onepose

    # for scene in glob.glob(data_root+'/*'):
    for scene_name in ['0447-nabati-box', '0450-hlychocpie-box', 
            '0488-jijiantoothpaste-box', '0493-haochidianeggroll-box', 
            '0494-qvduoduocookies-box', '0594-martinBootsLeft-others']:
        scene = os.path.join(data_root, scene_name)
        print(scene)
        process_scene(scene)

        # scene_name = os.path.basename(scene)
        copy_sfm_model(
            os.path.join(sfm_data_root, scene_name, 'outputs_superpoint_superglue/sfm_ws/model'), 
            os.path.join(scene, 'colmap/sparse')
        )
        shutil.copyfile(
            os.path.join(sfm_data_root, scene_name, 'outputs_superpoint_superglue/model.ply'),
            os.path.join(scene, 'pc.ply')
        )
        run_colmap_mvs(os.path.join(scene, 'colmap/sparse'), scene, os.path.join(scene, 'colmap/dense'))
