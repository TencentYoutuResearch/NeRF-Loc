"""
Author: jenningsliu
Date: 2022-04-27 10:02:03
LastEditors: jenningsliu
LastEditTime: 2022-04-27 10:39:51
FilePath: /nerf-loc/datasets/colmap/cli.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import sys
import subprocess

def run_colmap_mvs(sparse_path, image_path, dense_path):
    cmd = [
        'colmap image_undistorter',
        f'--image_path {image_path}',
        f'--input_path {sparse_path}',
        f'--output_path {dense_path}',
        f'--output_type COLMAP',
        f'--max_image_size 2000'
    ]
    out = subprocess.run(' '.join(cmd), capture_output=False, shell=True, check=False)
    if out.returncode != 0:
        print('Run colmap image_undistorter failed!')
        sys.exit(1)

    cmd = [
        'colmap patch_match_stereo',
        f'--workspace_path {dense_path}',
        f'--workspace_format COLMAP',
        f'--PatchMatchStereo.geom_consistency true'
    ]
    out = subprocess.run(' '.join(cmd), capture_output=False, shell=True, check=False)
    if out.returncode != 0:
        print('Run colmap patch_match_stereo failed!')
        sys.exit(1)
