"""
Author: jenningsliu
Date: 2022-04-11 16:17:04
LastEditors: jenningsliu
LastEditTime: 2022-08-24 12:35:34
FilePath: /nerf-loc/models/pose_optimizer.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import numpy as np

from utils.transform.se3 import se3_exp_map, se3_log_map
from .superpoint import SuperPoint

from utils.metrics import compute_pose_error

class PoseOptimizer(object):
    def __init__(self, args, nerf_renderer, debug=False, sampling='random', use_feat=True):
        super().__init__()
        self.args = args
        self.sampling = sampling
        self.nerf_renderer = nerf_renderer
        self.debug = debug
        self.use_feat = use_feat

        if self.sampling == 'interest_region':
            self.sp = SuperPoint({
                'nms_radius': 10,
                'keypoint_threshold':0.005,
                'max_keypoints': 1024,
                'model_path': '/apdcephfs/private_jenningsliu/SuperMatch/weights/superpoint_v1.pth'
            }).cuda().eval()

    def __call__(self, pose, data,
                max_steps=100, lr=0.001, scale=0.25):
        self.nerf_renderer = self.nerf_renderer.eval()

        K = data['K'].clone()
        image = data['img'].clone()
        H,W = image.shape[-2:]
        if scale is not None:
            K[:2] *= scale
            image = F.interpolate(
                image[None], size=(int(H * scale), int(W * scale)), mode='bilinear', align_corners=False)[0]
            H,W = image.shape[-2:]

        if not torch.is_tensor(pose):
            pose = torch.tensor(pose).to(image.device).float()

        if pose.shape == (3,4):
            bottom = torch.tensor([0,0,0,1]).view(1,4).to(pose.device)
            pose = torch.cat([pose, bottom])

        pose_gt = torch.clone(data['pose'])
        # ### add noise to gt as intial pose
        # if self.debug:
        #     noise = torch.randn(6).float().to(image.device) * 5e-2
        #     pose = torch.matmul(se3_exp_map(noise[None])[0], pose_gt)
        #     # pose = torch.matmul(se3_exp_map(noise[None])[0], pose)
        #     rot_err, trans_err = compute_pose_error(pose.cpu().numpy(), pose_gt.cpu().numpy())
        #     print('before debug: ', rot_err, trans_err, pose)
        # ###

        # init_pose = torch.tensor(pose).float().to(image.device)
        init_pose = pose
        # delta_pose = torch.randn(6).float().to(image.device) * 1e-6
        # delta_pose.requires_grad = True
        # optimizer = torch.optim.Adam(params=[delta_pose], lr=lr, betas=(0.9, 0.999))

        cam_pose_params = se3_log_map(torch.tensor(pose).float().to(image.device)[None])
        cam_pose_params.requires_grad = True
        optimizer = torch.optim.Adam(params=[cam_pose_params], lr=lr, betas=(0.9, 0.999))

        # cam_pose = torch.tensor(pose).float().to(image.device)
        # cam_pose.requires_grad = True
        # optimizer = torch.optim.Adam(params=[cam_pose], lr=lr, betas=(0.9, 0.999))

        rgb_target = image.permute(1,2,0)
        if self.use_feat:
            feat_target = F.interpolate(
                data['feat_pyramid']['layer1'], size=(H,W), mode='bilinear', align_corners=False
            ).permute(0,2,3,1)[0]

        if self.sampling == 'interest_region':
            gray = torchvision.transforms.functional.rgb_to_grayscale(image.unsqueeze(0))
            with torch.no_grad():
                keypoints = self.sp({'image': gray})['keypoints'][0].long()
            # print(f'keypoints: {keypoints.shape}')

            dialation = 1
            window_size = 3
            radius = dialation*window_size//2
            mask = torch.zeros((H, W))
            for pt in keypoints:
                u,v = pt
                mask[v-radius:v+radius+1:dialation, u-radius:u+radius+1:dialation] = 1
            # mask[pts2d[:,1],pts2d[:,0]] = 1
            # mask = mask.reshape(H*W)

            idxs_sampled = torch.where(mask.reshape(-1)>0)[0]
            v,u = torch.where(mask>0)
            uv_sampled = torch.stack([u,v], dim=1)
        elif self.sampling == 'grid':
            dialation = 10
            mask = torch.zeros((H, W))
            mask[::dialation, ::dialation] = 1
            idxs_sampled = torch.where(mask.reshape(-1)>0)[0]
            v,u = torch.where(mask>0)
            uv_sampled = torch.stack([u,v], dim=1)
        elif self.sampling == 'random':
            if 'target_mask' in data:
                v,u = torch.where(data['target_mask']>0)
                uv = torch.stack([u,v], dim=1)
            else:
                u, v = torch.meshgrid(torch.arange(W), torch.arange(H))
                u = u.reshape(-1).long()
                v = v.reshape(-1).long()
                uv = torch.stack([u,v], dim=1)
            idxs_sampled = np.random.choice(len(uv), 512, replace=False)
            uv_sampled = uv[idxs_sampled]
        else:
            raise NotImplementedError

        loss_init = None
        inter_poses = []
        with torch.enable_grad():
            for i_step in range(max_steps):
                # cam_pose = torch.matmul(se3_exp_map(delta_pose[None])[0].T, init_pose)
                cam_pose = se3_exp_map(cam_pose_params)[0]

                rays = self.nerf_renderer.points_2d_to_rays(uv_sampled, H, W, K, cam_pose)
                rays['depth_range'] = data['depth_range'][0]
                output = self.nerf_renderer.render_rays({
                    'pose': cam_pose,
                    'K': K,
                    'depth_range': data['depth_range'],
                    'feat_fine_src': data['feat_fine_src'],
                    'embedding_a': data['embedding_a'],
                    'topk_poses': data['topk_poses'],
                    'topk_Ks': data['topk_Ks'],
                    'topk_images': data['topk_images'],
                    'topk_depths': data['topk_depths'],
                }, rays)
                if self.use_feat:
                    loss = torch.mean(((
                        output['feat'] - feat_target[uv_sampled[:,1], uv_sampled[:,0]]
                    ) * output['mask'].unsqueeze(1)) ** 2)
                else:
                    loss = torch.mean(((
                        output['rgb'] - rgb_target[uv_sampled[:,1], uv_sampled[:,0]]
                    ) * output['mask'].unsqueeze(1)) ** 2)
                if loss.isnan().any():
                    return init_pose
                if loss_init is None:
                    loss_init = loss
                if i_step % 10 == 0 and self.debug:
                    print(f'loss: {loss}')
                    inter_poses.append(cam_pose)

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                # for param_group in optimizer.param_groups:
                #     param_group['lr'] = lr * (0.5 ** (i_step / 10))
                # print(f'loss: {loss}, lr: {lr * (0.5 ** (i_step / 10))}')

        if loss > loss_init:
            print('loss increase! discard pose optimize result.')
            return init_pose

        # print(delta_pose, se3_exp_map(delta_pose[None])[0])
        # cam_pose = torch.matmul(se3_exp_map(delta_pose[None])[0], init_pose).detach()
        cam_pose = se3_exp_map(cam_pose_params)[0]
        
        ###############
        if self.debug:
            rot_err, trans_err = compute_pose_error(cam_pose.cpu().numpy(), pose_gt.cpu().numpy())
            print('after debug: ', rot_err, trans_err, cam_pose.cpu().numpy())
        if self.debug:
            from PIL import Image
            gt_img = (image.permute(1,2,0).contiguous().cpu().numpy()*255).astype(np.uint8)
            # visualize overlay image
            for i, inter_pose in enumerate(inter_poses):
                data_cp = {k:v for k,v in data.items()}
                data_cp.update({'H': H, 'W': W, 'K': K, 'pose': inter_pose})
                ret = self.nerf_renderer.render_image(data_cp)
                render_img =( (ret['rgb']).cpu().numpy()*255).astype(np.uint8)
                overlay_img = (gt_img * 0.5 + render_img * 0.5).astype(np.uint8)
                Image.fromarray(np.concatenate([gt_img,render_img,overlay_img], axis=1)).save(f'vis_opt/{i}.png')
            from IPython import embed;embed()
        ###############
        return cam_pose
