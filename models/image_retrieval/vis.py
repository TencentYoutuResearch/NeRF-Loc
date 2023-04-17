"""
Author: jenningsliu
Date: 2022-07-01 18:13:03
LastEditors: jenningsliu
LastEditTime: 2022-07-13 10:31:46
FilePath: /nerf-loc/models/image_retrieval/vis.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import os
import sys
import cv2
import pickle
import numpy as np

if __name__ == '__main__':
    scene_path = sys.argv[1]
    split = sys.argv[2]
    method = sys.argv[3]

    with open(os.path.join(scene_path, f'image_retrieval_{split}_{method}.pkl'), 'rb') as f:
        data = pickle.load(f)
    
    for query, srcs in list(data.items())[::20]:
        query_img = cv2.imread(os.path.join(os.path.dirname(scene_path), query))
        src_imgs = []
        for src in srcs:
            src_img = cv2.imread(os.path.join(os.path.dirname(scene_path), src))
            src_imgs.append(src_img)
        cv2.imwrite('vis_retrieval.png', np.concatenate([query_img]+src_imgs[:8], axis=1))
        from IPython import embed;embed()
