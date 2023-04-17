'''
Author: jenningsliu
Date: 2022-03-10 17:46:34
LastEditors: jenningsliu
LastEditTime: 2022-08-15 14:24:41
FilePath: /nerf-loc/datasets/video/preprocess_cambridge.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
'''
import sys
import glob
import pickle as pkl
import os
from os import path as osp
import numpy as np
import math
import cv2
import trimesh
from tqdm import tqdm
import subprocess

from datasets.colmap.read_write_model import *
from datasets.colmap.cli import run_colmap_mvs
# from third_party.NeuRay.colmap.read_write_dense import read_array

'''
render depth maps are downloaded from https://github.com/vislearn/LessMore
scenes = [
	'https://www.repository.cam.ac.uk/bitstream/handle/1810/251342/KingsCollege.zip',
	'https://www.repository.cam.ac.uk/bitstream/handle/1810/251340/OldHospital.zip',
	'https://www.repository.cam.ac.uk/bitstream/handle/1810/251336/ShopFacade.zip',
	'https://www.repository.cam.ac.uk/bitstream/handle/1810/251294/StMarysChurch.zip',
	'https://www.repository.cam.ac.uk/bitstream/handle/1810/251291/GreatCourt.zip',
]
'''
MAX_DEPTH = 300

def load_reconstruction(recon_file):
    with open(recon_file, 'r') as f:
        reconstruction = f.readlines()

    num_cams = int(reconstruction[2])
    num_pts = int(reconstruction[num_cams + 4])

    # read points
    view_pts_dict = {}
    for cam_idx in range(0, num_cams):
        view_pts_dict[cam_idx] = []

    pt = pts_start = num_cams + 5
    pts_end = pts_start + num_pts

    pts_3d = []
    while pt < pts_end:

        pt_list = reconstruction[pt].split()
        xyz = [float(x) for x in pt_list[0:3]]
        # xyz.append(1.0)
        rgb = [int(x) for x in pt_list[3:6]]

        image_ids = []
        point2D_idxs = []
        for pt_view in range(0, int(pt_list[6])):
            image_id = int(pt_list[7 + pt_view * 4])
            # feature_id = int(pt_list[8 + pt_view * 4])
            feature_id = len(view_pts_dict[image_id])
            image_ids.append(image_id)
            point2D_idxs.append(feature_id)
            view_pts_dict[image_id].append({
                'point3D_id': pt,
                # 'point_3d': xyz,
                'point2D': [float(x) for x in pt_list[9+pt_view*4 : 11+pt_view*4]]
            })

        pts_3d.append({
            'id': pt,
            'xyz': np.array(xyz),
            'rgb': np.array(rgb),
            'image_ids': np.array(image_ids),
            'point2D_idxs': np.array(point2D_idxs)
        })
        
        pt += 1

    print('Reconstruction contains %d cameras and %d 3D points.' % (num_cams, num_pts))
    # pts_3d = np.array(pts_3d)[:,:3]
    return reconstruction, view_pts_dict, pts_3d


def parse_camera_pose(camera):
    cam_rot = [float(r) for r in camera[4:]]

    #quaternion to axis-angle
    angle = 2 * math.acos(cam_rot[0])
    x = cam_rot[1] / math.sqrt(1 - cam_rot[0]**2)
    y = cam_rot[2] / math.sqrt(1 - cam_rot[0]**2)
    z = cam_rot[3] / math.sqrt(1 - cam_rot[0]**2)

    cam_rot = [x * angle, y * angle, z * angle]
    
    cam_rot = np.asarray(cam_rot)
    cam_rot, _ = cv2.Rodrigues(cam_rot)

    cam_trans = [float(r) for r in camera[1:4]]
    cam_trans = np.asarray([cam_trans])
    cam_trans = np.transpose(cam_trans)
    cam_trans = - np.matmul(cam_rot, cam_trans)

    # if np.absolute(cam_trans).max() > 10000:
    #     print('Skipping image ' + image_file + '. Extremely large translation. Outlier?')
    #     print(cam_trans)
    #     continue

    cam_pose = np.concatenate((cam_rot, cam_trans), axis = 1)
    cam_pose = np.concatenate((cam_pose, [[0, 0, 0, 1]]), axis = 0)
    return cam_rot, cam_trans, cam_pose

def convert_to_colmap(reconstruction, pts_dict, pts_3d, scene_folder):
    num_cams = len(pts_dict)

    camera_list_all = []
    image_list_all = []
    with open(osp.join(scene_folder, f'dataset_train.txt'), 'r') as f:
        camera_list = f.readlines()
        camera_list = camera_list[3:]
        image_list = [camera.split()[0] for camera in camera_list]
        camera_list_all += camera_list
        image_list_all += image_list
    with open(osp.join(scene_folder, f'dataset_test.txt'), 'r') as f:
        camera_list = f.readlines()
        camera_list = camera_list[3:]
        image_list = [camera.split()[0] for camera in camera_list]
        camera_list_all += camera_list
        image_list_all += image_list

    cameras = {}
    images = {}
    points3D = {}
    for p3d in pts_3d:
        point3D_id = p3d['id']
        points3D[point3D_id] = Point3D(id=point3D_id, xyz=p3d['xyz'], rgb=p3d['rgb'],
                                        error=0, image_ids=p3d['image_ids'],
                                        point2D_idxs=p3d['point2D_idxs'])

    for cam_idx in tqdm(range(num_cams)):
        # if cam_idx % 50 != 0:
        #     continue
        # print('Processing camera %d of %d.' % (cam_idx, num_cams))
        image_file = reconstruction[3 + cam_idx].split()[0]
        image_file = image_file[:-3] + 'png'

        if image_file not in image_list_all:
            continue
        image_idx = image_list_all.index(image_file)

        # read camera
        # camera = line.strip('\n').split()
        camera = camera_list_all[image_idx].split()
        cam_rot, cam_trans, cam_pose = parse_camera_pose(camera)
        Tcw = cam_pose

        qvec = rotmat2qvec(Tcw[:3,:3])
        tvec = Tcw[:3,3]
        # image_id = cam_idx
        # img_name = osp.join(scene, camera[0])
        img_name = camera[0]
        # print(img_name)
        image_id = camera_id = cam_idx
        images[image_id] = Image(
            id=image_id, qvec=qvec, tvec=tvec,
            camera_id=camera_id, name=img_name,
            xys=np.array([p['point2D'] for p in pts_dict[image_id]]),
            point3D_ids=[p['point3D_id'] for p in pts_dict[image_id]]
        )

        focal_length = float(reconstruction[3 + cam_idx].split()[1])
        image = cv2.imread(osp.join(scene_folder, image_file))
        H,W = image.shape[:2]

        CAM_MODELS = {'SIMPLE_PINHOLE': 0,
                        'PINHOLE': 1,
                        'SIMPLE_RADIAL': 2,
                        'RADIAL': 3,
                        'OPENCV': 4,
                        'FULL_OPENCV': 5,
                        'SIMPLE_RADIAL_FISHEYE': 6,
                        'RADIAL_FISHEYE': 7,
                        'OPENCV_FISHEYE': 8,
                        'FOV': 9,
                        'THIN_PRISM_FISHEYE': 10}
        # CAM_MODELS_NAME = {v:k for k,v in CAM_MODELS.items()}
        cameras[camera_id] = Camera(
            id=camera_id,
            model='SIMPLE_PINHOLE',
            width=W,
            height=H,
            params=np.array([focal_length,W*0.5,H*0.5])
        )

    # # check data
    # all_images_id = list(images.keys())
    # all_pts_id = list(points3D.keys())
    # for p3d_id, p3d in points3D.items():
    #     assert len(p3d.image_ids) == len(p3d.point2D_idxs)
    #     for image_id, p2d_id in zip(p3d.image_ids, p3d.point2D_idxs):
    #         assert image_id in all_images_id
    #         assert p2d_id >= 0 and p2d_id < len(images[image_id].xys), f'{p2d_id} {len(images[image_id].xys)}'

    output_path = os.path.join(scene_folder, 'colmap/sparse')
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    write_model(cameras, images, points3D, path=output_path, ext='.bin')

def process_split(scene_folder, scene, split, reconstruction, pts_dict, pts_3d):
    meta_infos = []

    num_cams = len(pts_dict)
    xyz_world = np.array([pt['xyz'] for pt in pts_3d])

    with open(osp.join(scene_folder, f'dataset_{split}.txt'), 'r') as f:
        camera_list = f.readlines()
        camera_list = camera_list[3:]
        image_list = [camera.split()[0] for camera in camera_list]

        for cam_idx in tqdm(range(num_cams)):
            # print('Processing camera %d of %d.' % (cam_idx, num_cams))
            image_file = reconstruction[3 + cam_idx].split()[0]
            image_file = image_file[:-3] + 'png'

            if image_file not in image_list:
                # print('Skipping image ' + image_file + '. Not part of set: ' + split + '.')
                continue
            image_idx = image_list.index(image_file)

            # read camera
            # camera = line.strip('\n').split()
            camera = camera_list[image_idx].split()
            cam_rot, cam_trans, cam_pose = parse_camera_pose(camera)

            if np.absolute(cam_trans).max() > 10000:
                print('Skipping image ' + image_file + '. Extremely large translation. Outlier?')
                print(cam_trans)
                continue

            Tcw = cam_pose

            focal_length = float(reconstruction[3 + cam_idx].split()[1])

            image = cv2.imread(osp.join(scene_folder, image_file))
            H,W = image.shape[:2]
            img_name = camera[0]
            seq, frame = img_name.split('.')[0].split('/')
            # depth_file_name = osp.join(f'cambridge_rendered_depth/{scene}/{seq}_{frame}.depth.tiff')
            # if scene == 'ShopFacade':
            #     depth_file_name = depth_file_name.replace('.tiff', '.png')
            depth_file_name = osp.join(scene, f'colmap/dense/stereo/depth_maps/{img_name}.geometric.bin')
            if split=='train' and not osp.exists(osp.join(osp.dirname(scene_folder), depth_file_name)):
                print('skip train image without depth: ', osp.join(osp.dirname(scene_folder), depth_file_name))
                continue
            
            # # compute near far from depth map
            # if osp.exists(osp.join(osp.dirname(scene_folder), depth_file_name)):
            #     # depth = cv2.imread(osp.join(osp.dirname(scene_folder), depth_file_name), cv2.IMREAD_ANYDEPTH)
            #     # depth = depth.reshape(-1).astype(np.float32) / 1000
            #     # mask = (depth > 0) & (depth < 1000)
            #     depth = read_array(osp.join(osp.dirname(scene_folder), depth_file_name))
            #     mask = (depth > 0) & (depth < 500)
            #     depth = depth[mask]
            #     near = np.percentile(depth, 0.1)
            #     far = np.percentile(depth, 99.9)
            # else:
            #     near = None
            #     far = None

            # compute near far from 3d points
            R = Tcw[:3,:3]
            t = Tcw[:3,3]
            K = np.array([
                [focal_length, 0, float(W)/2],
                [0, focal_length, float(H)/2],
                [0,0,1]
            ])
            xyz_cam = (R @ xyz_world.T + t.reshape(3,1)).T
            uvz = K @ xyz_cam.T
            uv = uvz[:2] / uvz[2:]
            z = xyz_cam[:,2]
            valid_mask = (uv[0] >= 0) & (uv[1] >= 0) & (uv[0] < W) & (uv[1] < H) & (z > 0) & (z < MAX_DEPTH)
            if valid_mask.sum() == 0:
                print('Warining: skip bad pose')
                continue
            # near = z[valid_mask].min()
            # far = z[valid_mask].max()
            near = np.percentile(z[valid_mask], 0.1)
            far = np.percentile(z[valid_mask], 99.)

            meta_infos.append({
                'file_name': osp.join(scene, img_name),
                'frame_id': int(frame.replace('frame', '')),
                'sequence_id': seq,
                'depth_file_name': depth_file_name,
                # 'extrinsic_Tcw': np.linalg.inv(Twc)[:3],
                'extrinsic_Tcw': Tcw[:3],
                'camera_intrinsic': np.array([
                    focal_length, focal_length, float(W)/2, float(H)/2,   0.,   0.], dtype=np.float32),
                'frame_dim': (H,W),
                'camera_index': cam_idx,
                'far': far,
                'near': near
            })
            # if 'seq1/frame00001' in img_name:
            #     print(meta_infos[-1]['extrinsic_Tcw'], meta_infos[-1]['camera_intrinsic'])
            #     from IPython import embed;embed()
    if split == 'train':
        print('near: ', np.array([m['near'] for m in meta_infos]).min())
        print('far: ', np.array([m['far'] for m in meta_infos]).max())
    with open(osp.join(scene_folder, f'info_{split}.pkl'), 'wb') as fout:
        pkl.dump(meta_infos, fout)
    return meta_infos

if __name__ == '__main__':
    data_root = sys.argv[1]
    for scene in ['KingsCollege','OldHospital','GreatCourt','ShopFacade','StMarysChurch']:
        print(scene)
        scene_folder = osp.join(data_root, scene)
        reconstruction, pts_dict, pts_3d = load_reconstruction(osp.join(scene_folder, 'reconstruction.nvm'))

        convert_to_colmap(reconstruction, pts_dict, pts_3d, scene_folder)
        run_colmap_mvs(
            os.path.join(scene_folder, 'colmap/sparse'), 
            scene_folder, os.path.join(scene_folder, 'colmap/dense'))
        
        train_meta_infos = process_split(scene_folder, scene, 'train', reconstruction, pts_dict, pts_3d)
        test_meta_infos = process_split(scene_folder, scene, 'test', reconstruction, pts_dict, pts_3d)

        # filter out extremely far points
        poses = np.array([
            np.linalg.inv(np.concatenate([i['extrinsic_Tcw'], np.array([[0,0,0,1]])]))[:3,3] for i in train_meta_infos
        ])
        avg_pose = poses.mean(0)
        xyz = np.array([p['xyz'] for p in pts_3d])
        rgb = np.array([p['rgb'] for p in pts_3d])
        offset = np.linalg.norm(xyz - avg_pose.reshape(1,3), axis=1)
        cloud = trimesh.PointCloud(vertices=xyz[offset<MAX_DEPTH], colors=rgb[offset<MAX_DEPTH])
        # cloud = trimesh.PointCloud(vertices=xyz, colors=rgb)
        print(f'After filtering, {len(cloud.vertices)} 3D points left.')
        cloud.export(os.path.join(scene_folder, 'pc.ply'))
