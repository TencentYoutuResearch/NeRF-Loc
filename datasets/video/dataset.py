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

import random
import numpy as np
from .image import *
from .reader import load_one_img, load_rgb_intrinsic
from .geometry import *
from . import transform
from .furthest_pose_sampler import FurtherPoseSampling
from .covisibility_sampler import CovisibilitySampling
from scipy.spatial import ConvexHull
import hashlib

def add_depth_offset(depth,mask,region_min,region_max,offset_min,offset_max,noise_ratio,depth_length):
    coords = np.stack(np.nonzero(mask), -1)[:, (1, 0)]
    length = np.max(coords, 0) - np.min(coords, 0)
    center = coords[np.random.randint(0, coords.shape[0])]
    lx, ly = np.random.uniform(region_min, region_max, 2) * length
    diff = coords - center[None, :]
    mask0 = np.abs(diff[:, 0]) < lx
    mask1 = np.abs(diff[:, 1]) < ly
    masked_coords = coords[mask0 & mask1]
    global_offset = np.random.uniform(offset_min, offset_max) * depth_length
    if np.random.random() < 0.5:
        global_offset = -global_offset
    local_offset = np.random.uniform(-noise_ratio, noise_ratio, masked_coords.shape[0]) * depth_length + global_offset
    depth[masked_coords[:, 1], masked_coords[:, 0]] += local_offset

class AttrDict(dict):
    def __getattr__(self, name):
        value = self[name]
        if isinstance(value, dict):
            value = AttrDict(value)
        return value

class VideoDataset(Dataset):
    def __init__(
        self, args, cfg, split, mode='test'
    ):
        self.args = args
        self.cfg = cfg
        self.root_dir = cfg.base_dir
        self.scene = cfg.scene
        self.scene_dir = os.path.join(cfg.base_dir, cfg.scene)
        self.tempo_interval = cfg.tempo_interval

        self.max_seq_len = 0

        train_seq_list_path = os.path.join(cfg.base_dir, cfg.scene, 'info_train.pkl')
        test_seq_list_path = os.path.join(cfg.base_dir, cfg.scene, 'info_test.pkl')
        self.train_meta_info_list, self.train_image_retrieval = \
            self.load_meta_info_list(cfg.base_dir, train_seq_list_path, args.image_retrieval_method_train)
        self.test_meta_info_list, self.test_image_retrieval = \
            self.load_meta_info_list(cfg.base_dir, test_seq_list_path, args.image_retrieval_method_test)

        self.train_sequences, self.train_name2sequence = \
            self.build_sequence_meta_info(self.train_meta_info_list) # seq_name:frames

        synthesis_seq_list_path = os.path.join(cfg.base_dir, cfg.scene, 'synthesis', 'info.pkl')
        if os.path.exists(synthesis_seq_list_path):
            synthesis_meta_info_list, synthesis_image_retrieval = \
                self.load_meta_info_list(cfg.base_dir, synthesis_seq_list_path, args.image_retrieval_method_train)
            # print(f'Found {len(synthesis_meta_info_list)} synthesis frames from {synthesis_seq_list_path}')
            self.synthesis_meta_info_list = synthesis_meta_info_list
            self.synthesis_image_retrieval = synthesis_image_retrieval
            # self.train_expand_meta_info_list = self.train_meta_info_list + synthesis_meta_info_list
            # self.train_meta_info_list = synthesis_meta_info_list
        else:
            self.synthesis_meta_info_list = []
            self.synthesis_image_retrieval = {}

        self.set_split(split)
        self.mode = mode

        transforms = []
        for t in cfg.TRANSFORM:
            name = list(t.keys())[0]
            params = list(t.values())[0]
            transforms.append(getattr(transform, name)(**params))
        self.transform = transform.Compose(transforms)


        aug_transforms = []
        for t in cfg.get('AUG_TRANSFORM', []):
            name = list(t.keys())[0]
            params = list(t.values())[0]
            aug_transforms.append(getattr(transform, name)(**params))
        self.aug_transform = transform.Compose(aug_transforms)

        if args.test_time_color_jitter:
            self.color_jitter = transform.ColorJitter(brightness=0.75, contrast=0.75, saturation=0.75, hue=0.1)
            self.test_time_color_jitter_params = [
                # (torch.tensor([0, 3, 2, 1]), 0.25, 0.5, 0.5, -0.1),
                # (torch.tensor([0, 3, 2, 1]), 0.5, 0.25, 0.5, 0.1),
                # (torch.tensor([0, 1, 2, 3]), 0.75, 1.75, 0.25, 0.2),
                # (torch.tensor([3, 1, 2, 0]), 1.5, 1.5, 1.75, 0.1),
                # (torch.tensor([3, 1, 2, 0]), 1.5, 1.5, 0.9, 0.05),
            ]
            for brightness in [0.25,0.75,1.25,1.75]:
                for contrast in [0.25,0.75,1.25,1.75]:
                    for saturation in [0.25,0.75,1.25,1.75]:
                        for hue in [0.05,-0.05]:
                            self.test_time_color_jitter_params.append(
                                (torch.tensor([0, 1, 2, 3]), brightness, contrast, saturation, hue))
            # self.color_jitter.set_parameters((torch.tensor([0, 3, 2, 1]), 0.25, 0.5, 0.5, -0.1))
            # print('test_time_color_jitter params:', self.color_jitter.params)
        
        # to filter point cloud outsize RoI
        bboxes_3d_path = os.path.join(cfg.base_dir, cfg.scene, 'bboxes_3d.npy')
        if os.path.exists(bboxes_3d_path):
            self.bboxes_3d = np.load(bboxes_3d_path)[:,:8,:] # (N,8,3)
        else:
            self.bboxes_3d = None

        if self.bboxes_3d is None:
            self.pc_path = pc_path = os.path.join(cfg.base_dir, cfg.scene, 'pc.ply')
        else:
            self.pc_path = pc_path = os.path.join(cfg.base_dir, cfg.scene, 'in_box_pc.ply')
        self.pc = None
        self.pc_range = None
        if os.path.exists(pc_path):
            self.pc = trimesh.load(pc_path)
            self.pc_range = np.concatenate([self.pc.vertices.min(0), self.pc.vertices.max(0)])
            # print(f'original pc range: {self.pc_range}')

        self.kp_idx = None
        kp_idx_path = os.path.join(cfg.base_dir, cfg.scene, 'model_keypoints_idx.npy')
        if os.path.exists(kp_idx_path):
            self.kp_idx = np.load(kp_idx_path)

        if 'near' in cfg and 'far' in cfg:
            self.near = cfg.near
            self.far = cfg.far
        else:
            self.near = self.train_meta_info_list[0]['near']
            self.far = self.train_meta_info_list[0]['far']
            for info in self.train_meta_info_list:
                self.near = min(self.near, info['near'])
                self.far = max(self.far, info['far'])

        # move scene origin to pose center
        train_Twc = np.array([
            np.linalg.inv(np.concatenate([m['extrinsic_Tcw'], np.array([0,0,0,1]).reshape(1,4)], axis=0)) \
                for m in self.train_meta_info_list[::self.tempo_interval]
        ])
        pose_center = train_Twc[:,:3,3].mean(0)
        center_T = np.eye(4)
        center_T[:3,3] = -pose_center
        self.transform_scene(center_T)
        self.scene_transform_matrix = center_T
        # print(f'move scene origin to {pose_center}')
        ########

        # scale_factor of the original scene
        if 'scale_factor' in cfg:
            self.scale_factor = cfg['scale_factor']
        elif 'rescale_far_limit' in cfg:
            assert cfg['rescale_far_limit'] > 0
             # so that the max far is scaled to cfg['rescale_far_limit']
            self.scale_factor = float(cfg['rescale_far_limit']) / self.far
        else:
            self.scale_factor = 1


        self.scale_scene(self.scale_factor)
        # print(f'rescale: {self.scale_factor}, near: {self.near}, far: {self.far}, pc range: {self.pc_range}')

        self.ref_poses = {d['file_name']:d['extrinsic_Tcw'] for d in self.train_meta_info_list[::self.tempo_interval]}
        self.ref_intrinsics = {
            d['file_name']:d['camera_intrinsic'] for d in self.train_meta_info_list[::self.tempo_interval]
        }
        self.ref_image_idx = {
            d['file_name']:i for i,d in enumerate(self.train_meta_info_list[::self.tempo_interval])
        }

        if 'coreset' in self.args.support_image_selection:
            # self.image_core_set = random.choices(list(self.ref_poses.keys()), k=16)

            if self.args.coreset_sampler == 'covisibility':
                self.coreset_sampler = CovisibilitySampling(self)
            elif self.args.coreset_sampler == 'FPS':
                self.coreset_sampler = FurtherPoseSampling(self)
            else:
                raise NotImplementedError
            np.random.seed(666)
            coreset_image_names = self.coreset_sampler.sample(max_k=self.args.image_core_set_size)
            self.image_core_set = self.load_support_images(coreset_image_names)
        else:
            self.image_core_set = None

    def load_support_images(self, topk_image_names):
        topk_idxs = []
        for name in topk_image_names:
            topk_idxs.append(self.ref_image_idx[name])
        topk_idxs = np.array(topk_idxs).astype(np.int64)
        topk_images, topk_depths, topk_w2cs, topk_Ks = \
            self.load_topk_frames([self.train_meta_info_list[::self.tempo_interval][i] for i in topk_idxs])
        if len(topk_w2cs) > 0:
            bottom = np.tile(np.array([0,0,0,1]).reshape(1,1,4), [len(topk_w2cs),1,1])
            topk_w2cs = np.concatenate([topk_w2cs, bottom], axis=1)
            topk_poses = np.linalg.inv(topk_w2cs).astype(np.float32) # c2w, [K,4,4]
        else:
            topk_poses = []
        return topk_idxs, topk_images, topk_depths, topk_poses, topk_Ks

    def load_mvs_support_images(self, topk_image_names):
        if len(topk_image_names) == 0:
            return self.load_support_images(topk_image_names)
        # if self.mode == 'train':
        #     nearest_name = random.choice(topk_image_names)
        # else:
        #     nearest_name = topk_image_names[0]
        nearest_name = topk_image_names[0]
        # seq_name = nearest_name.split('/')[-2]
        seq_name = self.train_name2sequence[nearest_name]
        seq = self.train_sequences[seq_name]
        frame_names = [x['file_name'] for x in seq]
        idx = frame_names.index(nearest_name)
        step = 5
        if idx < step:
            support_idxs = [idx + step, idx + 2*step]
        elif idx >= len(seq) - step:
            support_idxs = [idx - step, idx - 2*step]
        else:
            support_idxs = [idx - step, idx + step]
        support_image_names = [nearest_name] + [frame_names[i] for i in support_idxs]
        # from IPython import embed;embed()
        return self.load_support_images(support_image_names)

    def compute_random_synthesis_range(self):
        t_max = np.ones(3) * self.args.synthesis_t_noise * self.scale_factor
        r_max = np.ones(3) * self.args.synthesis_r_noise
        poses = np.array([
            np.linalg.inv(np.concatenate([p, np.array([[0,0,0,1]])])) for p in self.ref_poses.values()
        ]) # camera to world
        centers = poses[:,:3,3]
        std = np.std(centers, axis=0)
        axis_scale = std / std.max()
        t_max = t_max * axis_scale
        # gravity_axis = std.argmin()
        # t_max[gravity_axis] = std[gravity_axis] # avoid pose that are lower than ground
        return t_max, r_max

    def scale_scene(self, scale_factor=1.):
        for meta in self.train_meta_info_list:
            meta['extrinsic_Tcw'][:,3] *= scale_factor
            meta['near'] *= scale_factor
            meta['far'] *= scale_factor
        for meta in self.test_meta_info_list:
            meta['extrinsic_Tcw'][:,3] *= scale_factor
            # TODO: make sure exist
            if meta.get('near', None) is None:
                meta['near'] = self.near
            if meta.get('far', None) is None:
                meta['far'] = self.far
            meta['near'] *= scale_factor
            meta['far'] *= scale_factor
        if hasattr(self, 'synthesis_meta_info_list'):
            for meta in self.synthesis_meta_info_list:
                meta['extrinsic_Tcw'][:,3] *= scale_factor
                # TODO: make sure exist
                if meta.get('near', None) is None:
                    meta['near'] = self.near
                if meta.get('far', None) is None:
                    meta['far'] = self.far
                meta['near'] *= scale_factor
                meta['far'] *= scale_factor
        self.scale_factor = scale_factor
        self.near *= scale_factor
        self.far *= scale_factor
        if self.pc is not None:
            self.pc.vertices *= scale_factor
            self.pc_range *= scale_factor
        if self.bboxes_3d is not None:
            self.bboxes_3d *= scale_factor

    def transform_scene(self, T):
        """ Apply rigid transformation to the scene
        Args: 
            T: (4,4) rigid transformation applied to world points
        Returns: 
        """       
        T_inv = np.linalg.inv(T)
        for meta in self.train_meta_info_list:
            Tcw = np.concatenate([meta['extrinsic_Tcw'], np.array([0,0,0,1]).reshape(1,4)])
            meta['extrinsic_Tcw'] = (Tcw @ T_inv)[:3]
        for meta in self.test_meta_info_list:
            Tcw = np.concatenate([meta['extrinsic_Tcw'], np.array([0,0,0,1]).reshape(1,4)])
            meta['extrinsic_Tcw'] = (Tcw @ T_inv)[:3]
        if hasattr(self, 'synthesis_meta_info_list'):
            for meta in self.synthesis_meta_info_list:
                Tcw = np.concatenate([meta['extrinsic_Tcw'], np.array([0,0,0,1]).reshape(1,4)])
                meta['extrinsic_Tcw'] = (Tcw @ T_inv)[:3]
        if self.pc is not None:
            xyz = self.pc.vertices
            xyz_hom = np.concatenate([xyz, np.ones_like(xyz[:,:1])], axis=1)
            self.pc.vertices = (T @ xyz_hom.T)[:3].T
            self.pc_range = np.concatenate([self.pc.vertices.min(0), self.pc.vertices.max(0)])
        if self.bboxes_3d is not None:
            # n_boxes = len(self.bboxes_3d)
            xyz = self.bboxes_3d.reshape(-1,3)
            xyz_hom = np.concatenate([xyz, np.ones_like(xyz[:,:1])], axis=1)
            self.bboxes_3d = (T @ xyz_hom.T)[:3].T.reshape(-1,8,3)
        return

    def set_split(self, split):
        self.split = split
        if split == 'train':
            self.meta_info_list = self.train_meta_info_list
        elif split == 'train+synthesis':
            self.meta_info_list = self.train_meta_info_list + self.synthesis_meta_info_list
        elif split == 'synthesis':
            self.meta_info_list = self.synthesis_meta_info_list
        elif split == 'test':
            self.meta_info_list = self.test_meta_info_list
        # elif split == 'all':
        #     self.meta_info_list = self.train_meta_info_list + self.test_meta_info_list
        # # always use all data for testing
        # if split != 'test':
        #     self.meta_info_list = self.meta_info_list[::self.tempo_interval]
        self.meta_info_list = self.meta_info_list[::self.tempo_interval]

    def set_mode(self, mode):
        self.mode = mode

    def load_meta_info_list(self, base_dir, meta_path, image_retrieval_method):
        image_retrieval_path = \
            meta_path.replace('info', 'image_retrieval').replace('.pkl', f'_{image_retrieval_method}.pkl')
        if not os.path.exists(image_retrieval_path):
            # print(f'Warning: {image_retrieval_path} not exists!')
            image_retrieval = {}
        else:
            with open(image_retrieval_path, 'rb') as f:
                image_retrieval = pkl.load(f)
        with open(meta_path, "rb") as f:
            meta_info = pkl.load(f)
            for frame in meta_info:
                frame['base_dir'] = base_dir
                frame['top_k'] = image_retrieval.get(frame['file_name'], [])
                # frame['top_k'] = image_retrieval[frame['file_name']]
        return meta_info, image_retrieval

    def build_sequence_meta_info(self, meta_info):
        sequences = defaultdict(list)
        name2sequence = {}
        for frame in meta_info:
            seq_name = frame['sequence_id']
            sequences[seq_name].append(frame)
            name2sequence[frame['file_name']] = seq_name
        for seq_name in sequences:
            sequences[seq_name] = sorted(sequences[seq_name], key=lambda x: x['frame_id'])
        return sequences, name2sequence

    def valid_depth_ratio(self, depth):
        mask = depth > 1e-5
        ratio = mask.sum() / mask.size
        return ratio

    def to_blender_pose(self, Tcw):
        T_blender = np.eye(4)
        T_blender[1,1] = -1
        T_blender[2,2] = -1
        if Tcw.shape[0] == 3:
            Tcw = np.concatenate([
                Tcw,
                np.array([0,0,0,1]).reshape(1,4)
            ])
        Tcw_blender = np.linalg.inv(T_blender) @ Tcw # world to blender camera frame
        Twc_blender = np.linalg.inv(Tcw_blender).astype(np.float32)
        return Twc_blender

    def load_topk_frames(self, meta_infos):
        imgs = []
        depths = []
        Tcws = []
        Ks = []
        for meta_info in meta_infos:
            img, depth, Tcw, K = load_one_img(meta_info['base_dir'], meta_info)
            img, depth, Tcw, K, _ = self.transform(img, depth, Tcw, K)
            imgs.append(img.astype(np.float32).transpose(2,0,1) / 255.)
            depths.append(depth.astype(np.float32))
            Tcws.append(Tcw)
            Ks.append(K.astype(np.float32))
        imgs = np.array(imgs)
        Tcws = np.array(Tcws)
        depths = np.array(depths)
        Ks = np.array(Ks)
        return imgs, depths, Tcws, Ks

    def load_frame(self, meta_info, base_dir):
        # self.transform.random_parameters()

        # if last_name is not None:
        #     a = meta_info["file_name"].split("/")[:2]
        #     if a[0] != last_name[0] or a[1] != last_name[1]:
        #         print(a, last_name)

        # last_name = meta_info["file_name"].split("/")[:2]
        if self.mode == 'test' and self.args.test_time_style_change:
            # replace with styled image
            print(
                'use style image: ', 
                meta_info["file_name"].replace('/seq', '/style_images/seq').replace('/frame', '_night/frame')
            )
            meta_info = copy.deepcopy(meta_info)
            meta_info["file_name"] = \
                meta_info["file_name"].replace('/seq', '/style_images/seq').replace('/frame', '_night/frame')

        img, depth, Tcw, K = load_one_img(
            base_dir,
            meta_info,
        )

        # compute target mask
        target_mask = None
        if self.bboxes_3d is not None:
            Twc = np.eye(4)
            Twc[:3] = Tcw[:3]
            Twc = np.linalg.inv(Twc)
            # xyz = self.bboxes_3d.reshape(-1,3)
            xyz = np.array(self.pc.vertices)
            target_mask = \
                self.compute_target_mask(xyz, K, Twc, depth.astype(np.float32)*self.scale_factor).astype(np.uint8)

        img, depth, Tcw, K, target_mask = self.transform(img, depth, Tcw, K, mask=target_mask)

        if self.mode == 'train':
            self.aug_transform.random_parameters()
            img, depth, Tcw, K, target_mask = self.aug_transform(img, depth, Tcw, K, target_mask)
            # img_tensor = self.aug_transform(torch.tensor(img_tensor)).numpy()
        if self.mode == 'test' and self.args.test_time_color_jitter:
            # test time color jitter param is randomly selected according to the hash of filename
            param_idx = int(hashlib.sha1(meta_info["file_name"].encode("utf-8")).hexdigest(), 16) \
                % len(self.test_time_color_jitter_params)
            print(meta_info["file_name"], param_idx, self.test_time_color_jitter_params[param_idx])
            self.color_jitter.set_parameters(self.test_time_color_jitter_params[param_idx])
            img, depth, Tcw, K, target_mask = self.color_jitter(img, depth, Tcw, K, target_mask)

        Twc = np.eye(4)
        Twc[:3] = Tcw[:3]
        Twc = np.linalg.inv(Twc)

        Twc = Twc.astype(np.float32)
        K_tensor = K.astype(np.float32)
        depth_tensor = depth.astype(np.float32)
        img_tensor = img.astype(np.float32).transpose(2,0,1) / 255.

        if self.args.support_image_selection == 'coreset':
            topk_idxs, topk_images, topk_depths, topk_poses, topk_Ks = \
                copy.deepcopy(self.image_core_set) # support images are fixed in this case
        elif self.args.support_image_selection == 'mvs':
            topk_idxs, topk_images, topk_depths, topk_poses, topk_Ks = self.load_mvs_support_images(meta_info['top_k'])
        elif self.args.support_image_selection == 'coreset+retrieval':
            core_idxs, core_images, core_depths, core_poses, core_Ks = copy.deepcopy(self.image_core_set)
            topk_idxs, topk_images, topk_depths, topk_poses, topk_Ks = \
                self.load_support_images(meta_info['top_k'][:-self.args.image_core_set_size])
            topk_idxs = np.concatenate([topk_idxs, core_idxs])
            topk_images = np.concatenate([topk_images, core_images])
            topk_depths = np.concatenate([topk_depths, core_depths])
            topk_poses = np.concatenate([topk_poses, core_poses])
            topk_Ks = np.concatenate([topk_Ks, core_Ks])
        else:
            topk_idxs, topk_images, topk_depths, topk_poses, topk_Ks = self.load_support_images(meta_info['top_k'])

        depth_tensor *= self.scale_factor
        topk_depths *= self.scale_factor
        result = {
            "filename": meta_info["file_name"],
            "depth_filename": os.path.join(self.root_dir, meta_info["depth_file_name"]),
            # "pose_": Twc[:3], # conventional camera to world
            # "pose": Twc_blender[:3], # blender camera to world
            "pose": Twc,
            "topk_poses": topk_poses, # poses of top-k similar images
            "topk_idxs": topk_idxs, # idx of top-k similar images
            'topk_images': topk_images,
            'topk_depths': topk_depths,
            'topk_Ks': topk_Ks,
            "K": K_tensor,
            "depth": depth_tensor,
            "image": img_tensor,
            # "near": max(meta_info['near'], 0.1), # per frame depth range
            # "far": meta_info['far'],
            "near": max(self.near, 0.01), # global depth range
            "far": self.far,
            "scene": self.scene,
            # "points3d": points3d,
            "scale_factor": self.scale_factor
        }
        if self.pc is not None:
            points3d = np.array(self.pc.vertices).astype(np.float32)
            if self.pc.colors is not None:
                points3d = np.concatenate([
                    points3d,
                    np.array(self.pc.colors[:,:3]).astype(np.float32)
                ], axis=1)
            if self.args.matching.keypoints_3d_sampling == 'response':
                keypoint3d_idx = self.kp_idx
                points3d = points3d[keypoint3d_idx]
            elif self.args.matching.keypoints_3d_sampling == 'random' \
                and self.args.matching.keypoints_3d_sampling_max_keep < len(points3d):
                keypoint3d_idx = \
                    np.random.choice(len(points3d), self.args.matching.keypoints_3d_sampling_max_keep, replace=False)
                points3d = points3d[keypoint3d_idx]

            result['points3d'] = points3d

        if self.mode == 'train' and self.cfg.get('aug_ref_depth', False):
            result['topk_depths_gt'] = copy.deepcopy(result['topk_depths'])
            result['topk_depths'] = \
                self.add_depth_noise(
                    result['topk_depths'], result['topk_depths'] > 0, [result['near'], result['far']]
                )
        # set depth outside range to zero
        for i, (idx, K,Twc,depth_tensor) in enumerate(zip(topk_idxs, topk_Ks, topk_poses, topk_depths)):
            depth_mask = (depth_tensor > result['near']) & (depth_tensor < result['far'])
            result['topk_depths'][i] *= depth_mask.astype(np.float32)

        # compute target mask
        if self.bboxes_3d is not None:
            result['bbox3d_corners'] = self.bboxes_3d.reshape(-1,3)
            # current frame
            result['target_mask'] = target_mask.astype(np.bool)
            # support frames
            i = 0
            for idx, K,Twc,depth_tensor in zip(topk_idxs, topk_Ks, topk_poses, topk_depths):
                target_mask = self.compute_target_mask(xyz, K, Twc, depth_tensor)
                result['topk_depths'][i] *= target_mask.astype(np.float32)
                i += 1
        return result

    def compute_target_mask(self, xyz, K, Twc, depth_tensor):
        uvz = self.get_projected_points(xyz, K, np.linalg.inv(Twc))
        hull = ConvexHull(uvz[:,:2])
        polygon = uvz[hull.vertices,:2].astype(np.int32)
        target_mask = np.zeros_like(depth_tensor).astype(np.uint8)
        cv2.fillPoly(target_mask, [polygon], 255)
        return target_mask > 0

    def get_projected_points(self, xyz, K, pose):
        xyz_hom = np.concatenate([xyz, np.ones_like(xyz[:,:1])], axis=1)
        xyz_cam = pose[:3,:3] @ xyz.T + pose[:3,3:]
        uvz = K @ xyz_cam
        uvz[:2] /= uvz[2:]
        return uvz.T

    def get_next_frame(self, idx, meta_info, idxs):
        start_idx = max(self.start_idxs[idx], idx - self.tempo_interval)
        return start_idx

    def crop_img(self, img, K, new_K):
        return img

    def add_depth_noise(self, depths, masks, depth_range):
        rfn = depths.shape[0]
        depths_output = []
        for rfi in range(rfn):
            depth, mask = depths[rfi], masks[rfi]
            depth = depth.copy()
            if mask.sum() == 0:
                depths_output.append(depth)
                continue
            near, far = depth_range
            depth_length = far - near
            if self.cfg['aug_use_depth_offset'] and np.random.random() < self.cfg['aug_depth_offset_prob']:
                add_depth_offset(depth, mask, self.cfg['aug_depth_offset_region_min'],
                                    self.cfg['aug_depth_offset_region_max'],
                                    self.cfg['aug_depth_offset_min'],
                                    self.cfg['aug_depth_offset_max'],
                                    self.cfg['aug_depth_offset_local'], depth_length)
            if self.cfg['aug_use_depth_small_offset'] \
                and np.random.random() < self.cfg['aug_depth_small_offset_prob']:
                add_depth_offset(depth, mask, 0.1, 0.2, 0.01, 0.05, 0.005, depth_length)
            if self.cfg['aug_use_global_noise'] and np.random.random() < self.cfg['aug_global_noise_prob']:
                depth += np.random.uniform(-0.005,0.005,depth.shape).astype(np.float32)*depth_length
            depths_output.append(depth)
        return np.asarray(depths_output)

    def __getitem__(self, idx):
        meta_info = self.meta_info_list[idx]

        data = self.load_frame(meta_info, meta_info["base_dir"])
        data['img_idx'] = idx

        return data

    def __len__(self):
        return len(self.meta_info_list)


def project_to_image(xyz, rgb, K, pose, image, out_name):
    xyz_cam = pose[:3,:3] @ xyz.T + pose[:3,3:]
    uvz = K@xyz_cam
    uv = uvz[:2] / uvz[2:]
    u = uv[0]
    v = uv[1]
    z = uvz[2]
    H,W = image.shape[:2]
    mask = (u >= 0) & (v >= 0) & (u < W) & (v < H) & (z > 0)
    sample_idx = np.random.choice(mask.sum(), 20000, replace=False)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    overlay = np.copy(image)
    # for pt,z,rgb in zip(uv.T[mask], z[mask], rgb[mask]):
    for pt,z,rgb in zip(uv.T[mask][sample_idx], z[mask][sample_idx], rgb[mask][sample_idx]):
        # color = tuple([int(c) for c in rgb[:3]])
        # color = (int(255*(1-min(1, z/2))), 0, int(255*min(1, z/2)))
        color = (0,0,255)
        cv2.circle(overlay, (int(pt[0]), int(pt[1])), 1, color, 1)
    # image[:,:,2] = render_image[:,:,0]
    # cv2.imwrite(out_name, image)
    image = cv2.addWeighted(image, 0.5, overlay, 0.5, 0)
    cv2.imwrite(out_name, overlay)

def blender_project_to_image(xyz, K, pose, image, out_name):
    K0 = np.eye(3,3)
    K0[1,1] = -1
    K0[2,2] = -1

    xyz_cam = pose[:3,:3] @ xyz.T + pose[:3,3:]
    xyz_cam_blender = K0@xyz_cam
    uvz = K@xyz_cam_blender
    uv = uvz[:2] / uvz[2:]
    u = uv[0]
    v = uv[1]
    z = uvz[2]
    H,W = image.shape[:2]
    mask = (u >= 0) & (v >= 0) & (u < W) & (v < H) & (z > 0)
    sample_idx = np.random.choice(mask.sum(), 20000, replace=False)
    for pt,d in zip(uv.T[mask][sample_idx], z[mask][sample_idx]):
        cv2.circle(image, (int(pt[0]), int(pt[1])), 1, (int(255*(1-d/5.)),0,int(255*(d/5.))), 1)
    cv2.imwrite(out_name, image)
