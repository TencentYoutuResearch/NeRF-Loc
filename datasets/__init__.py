"""
Author: jenningsliu
Date: 2022-02-28 19:38:14
LastEditors: jenningsliu
LastEditTime: 2022-08-23 17:13:51
FilePath: /nerf-loc/datasets/__init__.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import yaml 
from utils.common import AttrDict
import os
import copy

def build_dataset(args, split, phase='pose'):
    if args.dataset_type == 'blender':
        from .neuray_base_dataset import NeurayBaseDataset
        dataset = NeurayBaseDataset(args, split, f'nerf_synthetic/{args.scene}/white_400')
    elif args.dataset_type == 'llff':
        from .neuray_base_dataset import NeurayBaseDataset
        dataset = NeurayBaseDataset(args, split, f'llff_colmap/{args.scene}/low')
    elif args.dataset_type == 'colmap':
        from .colmap_dataset import ColmapDataset
        dataset = ColmapDataset(args, args.datadir, split, depth_type='colmap')
    elif args.dataset_type.startswith('video_'):
        from .video.dataset import VideoDataset
        from .video.multi_scene_dataset import MultiSceneDataset
        cfg_name = args.dataset_type.split('_')[1]
        cfg_file = f'configs/data/{cfg_name}.yaml'
        cfg = yaml.load(open(cfg_file), Loader=yaml.FullLoader)
        if phase == 'nerf':
            # no geometry augmentation during nerf training
            cfg['DATASET']['AUG_TRANSFORM'] = []
        if phase == 'nerf':
            # single scene
            cfg['DATASET']['scene'] = args.scene
            cfg = AttrDict(cfg)
            dataset = VideoDataset(args, cfg.DATASET, split)
        else:
            datasets = []
            # multiple scene
            for scene in args.scenes:
                scene_cfg = copy.deepcopy(cfg)
                scene_cfg['DATASET']['scene'] = scene
                scene_cfg = AttrDict(scene_cfg)
                ds = VideoDataset(args, scene_cfg.DATASET, split)
                datasets.append(ds)
            dataset = MultiSceneDataset(datasets)
    else:
        raise NotImplementedError
    return dataset
