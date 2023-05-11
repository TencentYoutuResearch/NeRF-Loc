"""
Author: jenningsliu
Date: 2022-03-25 17:55:43
LastEditors: jenningsliu
LastEditTime: 2022-07-14 10:48:40
FilePath: /nerf-loc/models/image_retrieval/run.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import sys
import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import cv2
import math
import pickle as pkl
from collections import defaultdict

import models.image_retrieval as image_retrieval
from models.image_retrieval.base_model import dynamic_load

from datasets import build_dataset
from utils.metrics import compute_pose_error

configs = {
    'dir': {
        'output': 'global-feats-dir',
        'model': {'name': 'dir'},
        'preprocessing': {'resize_max': 1024},
    },
    'netvlad': {
        'output': 'global-feats-netvlad',
        'model': {'name': 'netvlad'},
        'preprocessing': {'resize_max': 1024},
    }
}

def pairs_from_score_matrix(scores: torch.Tensor,
                            invalid: np.array,
                            num_select: int,
                            min_score: Optional[float] = None):
    assert scores.shape == invalid.shape
    if isinstance(scores, np.ndarray):
        scores = torch.from_numpy(scores)
    invalid = torch.from_numpy(invalid).to(scores.device)
    if min_score is not None:
        invalid |= scores < min_score
    scores.masked_fill_(invalid, float('-inf'))

    topk = torch.topk(scores, num_select, dim=1, sorted=True)
    indices = topk.indices.cpu().numpy()
    valid = topk.values.isfinite().cpu().numpy()

    pairs = []
    for i, j in zip(*np.where(valid)):
        pairs.append((i, indices[i, j]))
    return pairs

def retrieve_top_k(query_desc_dict, db_desc_dict, k=5, allow_self_match=False, interval=1):
    query_names = list(query_desc_dict.keys())
    db_names = list(db_desc_dict.keys())[::interval]
    print(f'db_image_size: {len(db_names)}')
    query_desc = torch.cat([query_desc_dict[n] for n in query_names], dim=0)
    db_desc = torch.cat([db_desc_dict[n] for n in db_names], dim=0)
    sim = torch.einsum('id,jd->ij', query_desc, db_desc)
    # Avoid self-matching
    if allow_self_match:
        invalid_matrix = np.zeros([len(query_names), len(db_names)], dtype=np.bool)
    else:
        invalid_matrix = np.array(query_names)[:, None] == np.array(db_names)[None]
    pairs = pairs_from_score_matrix(sim, invalid_matrix, k, min_score=0)
    top_k_pairs = defaultdict(list)
    for i,j in pairs:
        top_k_pairs[query_names[i]].append(db_names[j])
    # pairs = [(query_names[i], db_names[j]) for i, j in pairs]
    return top_k_pairs


def extract_global_descriptors(dataloader, model, device):
    descriptors = {}
    poses = {}
    bottom = np.array([[0,0,0,1]])
    for data in tqdm(dataloader):
        with torch.no_grad():
            ret = model({
                'image': data['image'].to(device)
            })
        # descriptors[data['filename']] = ret['global_descriptor'].cpu().numpy()
        descriptors[data['filename'][0]] = ret['global_descriptor']
        poses[data['filename'][0]] = np.concatenate([data['pose'][0].cpu().numpy(), bottom])
    return descriptors, poses


TINY_NUMBER = 1e-8
def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(np.clip(np.sum(vec1_unit*vec2_unit, axis=-1), -1.0, 1.0))
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    '''
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    '''
    assert R1.shape[-1] == 3 and R2.shape[-1] == 3 and R1.shape[-2] == 3 and R2.shape[-2] == 3
    return np.arccos(np.clip((np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1) / 2.,
                             a_min=-1 + TINY_NUMBER, a_max=1 - TINY_NUMBER))


def get_nearest_pose_ids(tar_pose, ref_poses, num_select, tar_id=-1, angular_dist_method='matrix',
                         scene_center=(0, 0, 0)):
    '''
    Args:
        tar_pose: target pose [3, 3]
        ref_poses: reference poses [N, 3, 3]
        num_select: the number of nearest views to select
    Returns: the selected indices
    '''
    num_cams = len(ref_poses)
    num_select = min(num_select, num_cams-1)
    batched_tar_pose = tar_pose[None, ...].repeat(num_cams, 0)

    if angular_dist_method == 'matrix':
        dists = batched_angular_dist_rot_matrix(batched_tar_pose[:, :3, :3], ref_poses[:, :3, :3])
    elif angular_dist_method == 'vector':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        scene_center = np.array(scene_center)[None, ...]
        tar_vectors = tar_cam_locs - scene_center
        ref_vectors = ref_cam_locs - scene_center
        dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
    elif angular_dist_method == 'dist':
        tar_cam_locs = batched_tar_pose[:, :3, 3]
        ref_cam_locs = ref_poses[:, :3, 3]
        dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
    else:
        raise Exception('unknown angular distance calculation method!')

    if tar_id >= 0:
        assert tar_id < num_cams
        dists[tar_id] = 1e3  # make sure not to select the target id itself

    sorted_ids = np.argsort(dists)
    selected_ids = sorted_ids[:num_select]
    # print(angular_dists[selected_ids] * 180 / np.pi)
    return selected_ids


def retrieve_top_k_oracle(query_pose_dict, db_pose_dict, k=5, allow_self_match=False, interval=1):
    query_names = list(query_pose_dict.keys())
    db_names = list(db_pose_dict.keys())[::interval]
    query_poses = torch.stack([torch.tensor(query_pose_dict[n]) for n in query_names], dim=0) # N,4,4
    query_translation = torch.stack([torch.tensor(query_pose_dict[n][:3,3]) for n in query_names], dim=0) # N,3
    db_poses = torch.stack([torch.tensor(db_pose_dict[n]) for n in db_names], dim=0) # M,4,4
    db_translation = torch.stack([torch.tensor(db_pose_dict[n][:3,3]) for n in db_names], dim=0) # M,3
    dists = torch.norm(query_translation.unsqueeze(1) - db_translation.unsqueeze(0), dim=-1)

    # pairs = pairs_from_score_matrix(-dists, invalid_matrix, 50, min_score=None)
    top_k_pairs = defaultdict(list)
    for i,query_name in enumerate(query_names):
        # Avoid self-matching
        if allow_self_match:
            tar_id = -1
        else:
            tar_id = db_names.index(query_name) if query_name in db_names else -1
        R_q = query_poses[i]
        ids = get_nearest_pose_ids(R_q.cpu().numpy(), db_poses.cpu().numpy(), len(db_poses), tar_id=tar_id)
        dist_mask = dists[i,ids] < 0.5
        if dist_mask.sum() == 0:
            dist_mask = dists[i,ids] < 0.75
        if dist_mask.sum() == 0:
            dist_mask = dists[i,ids] < 1.0
        if dist_mask.sum() == 0:
            dist_mask = dists[i,ids] < 1.5
        ids = ids[dist_mask]
        # assert len(ids) >= k, 'not enough near train images < 0.5'
        if len(ids) < k:
            print('not enough near train images < 0.5', len(ids))
        # rank_idx = np.argsort(dists[i, ids])
        # ids = ids[rank_idx]
        top_k_pairs[query_names[i]] = []
        for j in ids[:k]:
            top_k_pairs[query_names[i]].append(db_names[j])

    for name, value in top_k_pairs.items():
        assert len(value) > 0
    # pairs = [(query_names[i], db_names[j]) for i, j in pairs]
    return top_k_pairs

def extract_poses(dataloader):
    poses = {}
    bottom = np.array([[0,0,0,1]])
    for data in tqdm(dataloader):
        poses[data['filename'][0]] = np.concatenate([data['pose'][0].cpu().numpy(), bottom])
        # if len(poses) > 100:
        #     break
    return poses

def evaluate_image_retrieval(db_poses_dict, query_poses_dict, top_k_pairs, rot_thresh=30, trans_thresh=0.5, max_k=5):
    correct_count = []
    for query_name, top_k_db_names in top_k_pairs.items():
        rot_errs = []
        trans_errs = []
        for k in range(len(top_k_db_names)):
            if k > max_k:
                break
            rot_err, trans_err = compute_pose_error(query_poses_dict[query_name], db_poses_dict[top_k_db_names[k]])
            rot_errs.append(rot_err)
            trans_errs.append(trans_err)
        rot_errs = np.array(rot_errs)
        trans_errs = np.array(trans_errs)
        # print(rot_errs, trans_errs)
        correct_count.append(((rot_errs < rot_thresh) & (trans_errs < trans_thresh)).sum())
    correct_count = np.array(correct_count)
    recall = (correct_count > 0).mean()
    recall_2 = (correct_count > 2).mean()
    metrics = {}
    for min_correct_num in [1,2]:
        metrics[f'Avg-Recall@{rot_thresh}_{trans_thresh}_{min_correct_num}'] = \
            (correct_count >= min_correct_num).mean()
    print(f'Image retrieval metrics: ', metrics)

if __name__ == '__main__':
    import argparse
    from configs import get_cfg_defaults
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)

    k = cfg.image_core_set_size
    train_interval = cfg.image_retrieval_interval_train
    test_interval = cfg.image_retrieval_interval_test
    multi_trainset = build_dataset(cfg, 'train')
    multi_testset = build_dataset(cfg, 'test')
    # assert len(trainset.datasets) == 1 and len(testset.datasets) == 1
    for i in range(len(multi_trainset.datasets)):
        trainset = multi_trainset.datasets[i]
        testset = multi_testset.datasets[i]
        assert trainset.scene == testset.scene
        print(f'Running image retrieval for {trainset.scene} ' + 
            f'top-k: {k} train_interval: {train_interval} test_interval: {test_interval}')
        train_dataloader = torch.utils.data.DataLoader(
            trainset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)
        test_dataloader = torch.utils.data.DataLoader(
            testset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

        if cfg.image_retrieval_method == 'oracle':
            train_poses = extract_poses(train_dataloader)
            test_poses = extract_poses(test_dataloader)
            train_to_train = \
                retrieve_top_k_oracle(train_poses, train_poses, allow_self_match=False, k=k, interval=train_interval)
            test_to_train = \
                retrieve_top_k_oracle(test_poses, train_poses, allow_self_match=False, k=k, interval=test_interval)
        else:
            # conf = configs['dir']
            conf = configs[cfg.image_retrieval_method]
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            Model = dynamic_load(image_retrieval, conf['model']['name'])
            print(Model)
            model = Model(conf['model']).eval().to(device)
            
            train_descriptors, train_poses = extract_global_descriptors(train_dataloader, model, device=device)
            test_descriptors, test_poses = extract_global_descriptors(test_dataloader, model, device=device)

            train_to_train = retrieve_top_k(
                train_descriptors, train_descriptors, allow_self_match=False, k=k, interval=train_interval)
            test_to_train = retrieve_top_k(
                test_descriptors, train_descriptors, allow_self_match=False, k=k, interval=test_interval)

        evaluate_image_retrieval(train_poses, test_poses, test_to_train, max_k=k)

        with open(os.path.join(
                trainset.scene_dir, f'image_retrieval_train_{cfg.image_retrieval_method}.pkl'), 'wb') as f:
            pkl.dump(train_to_train, f)

        with open(os.path.join(
                testset.scene_dir, f'image_retrieval_test_{cfg.image_retrieval_method}.pkl'), 'wb') as f:
            pkl.dump(test_to_train, f)

        # if os.path.exists(os.path.join(trainset.root_dir, trainset.scene, 'synthesis', 'info.pkl')):
        #     # synthesis split
        #     synset = build_dataset(cfg, 'synthesis')
        #     syn_descriptors = extract_global_descriptors(synset, model)
        #     syn_to_train = retrieve_top_k(syn_descriptors, train_descriptors)

        #     with open(os.path.join(synset.scene_dir, 'synthesis/image_retrieval.pkl'), 'wb') as f:
        #         pkl.dump(syn_to_train, f)
