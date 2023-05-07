"""
Author: jenningsliu
Date: 2022-08-01 14:53:46
LastEditors: jenningsliu
LastEditTime: 2022-08-19 16:18:20
FilePath: /nerf-loc/utils/visualization.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import os
import cv2
import numpy as np
import imageio

def project_3d_points(pts3d, w2c, K):
    pts3d_cam = w2c[:3,:3] @ pts3d.T + w2c[:3,3:4]
    uvz = K @ pts3d_cam
    z = uvz[2].copy()
    uvz /= z[None]
    return uvz[:2].T, z

def draw_onepose_3d_box(box_corners, img, w2c, K, radius=5, color=(255, 255, 255)):
    """
    box_corners: [8,3]
    """
    arranged_points,z = project_3d_points(box_corners, w2c, K)
    EDGES = [
      [1, 5], [2, 6], [3, 7], [0,4],
      [1, 2], [5, 6], [4,7], [0,3],
      [1,0], [5,4], [6,7], [2,3]
    ] 
    for i in range(arranged_points.shape[0]):
        x, y = arranged_points[i]
        cv2.circle(
            img,
            (int(x), int(y)), 
            radius,
            color,
            -10
        )
    for edge in EDGES:
        start_points = arranged_points[edge[0]]
        start_x = int(start_points[0])
        start_y = int(start_points[1])
        end_points = arranged_points[edge[1]]
        end_x = int(end_points[0])
        end_y = int(end_points[1])
        cv2.line(img, (start_x, start_y), (end_x, end_y), color, 2)
    return img

def images_to_video(image_folder, video_save_path):
    imgs = []
    img_paths = glob.glob(image_folder+'/*')
    print(image_folder)
    print(img_paths[0])
    # img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[0]))
    # img_paths = sorted(img_paths, key=lambda x: int(os.path.basename(x).split('.')[-2].split('_')[-1]))
    img_paths = sorted(img_paths)
    for img_path in img_paths:
        print(img_path)
        imgs.append(cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_RGB2BGR))
    imageio.mimwrite(video_save_path, imgs, fps=30, quality=8)

if __name__ == '__main__':
    # from configs import config_parser
    # from datasets import build_dataset
    # parser = config_parser()
    # args = parser.parse_args()

    # multi_dataset = build_dataset(args, 'test')
    # dataset = multi_dataset.datasets[0]

    # data = dataset[0]

    # image = (data['image'].transpose(1,2,0)*255).astype(np.uint8)
    # K = data['K']
    # w2c = np.linalg.inv(data['pose'])
    # box_corners = dataset.bboxes_3d.reshape(-1,3)
    # image = draw_onepose_3d_box(box_corners, image, w2c, K)
    # cv2.imwrite('vis_box_3d.png', image)

    import sys
    import glob
    images_to_video(sys.argv[1], sys.argv[2])
