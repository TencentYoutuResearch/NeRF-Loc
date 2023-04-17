"""
Author: jenningsliu
Date: 2022-03-01 14:33:19
LastEditors: jenningsliu
LastEditTime: 2022-04-08 19:38:03
FilePath: /nerf-loc/utils/metrics.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import cv2
import numpy as np
import math

def compute_pose_error(T_est, T_gt):
    """ assuming two translation matrix are in the same scale
    Args: 
        T_est: np.array (4,4)
        T_gt: np.array (4,4)
    Returns: 
        angular_err: in degrees
        translatiton_err:
    """    
    r1 = T_est[:3,:3]
    r2 = T_gt[:3,:3]
    rot_diff= r2 @ r1.T
    trace = cv2.trace(rot_diff)[0]
    trace = min(3.0, max(-1.0, trace))
    angular_err = 180*math.acos((trace-1.0)/2.0)/np.pi

    t1 = T_est[:3,3]
    t2 = T_gt[:3,3]
    translatiton_err = np.linalg.norm(t1-t2)
    return angular_err, translatiton_err


def compute_matching_iou(pairs, pairs_gt):
    pred = zip(pairs[0].cpu().numpy().tolist(), pairs[1].cpu().numpy().tolist())
    gt = zip(pairs_gt[0].cpu().numpy().tolist(), pairs_gt[1].cpu().numpy().tolist())
    pred = set(pred)
    gt = set(gt)
    return len(pred.intersection(gt)) / (len(pred.union(gt))+1e-8)
