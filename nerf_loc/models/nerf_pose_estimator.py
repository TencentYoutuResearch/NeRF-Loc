'''
Author: jenningsliu
Date: 2022-02-25 16:44:28
LastEditors: jenningsliu jenningsliu@tencent.com
LastEditTime: 2022-06-29 22:45:54
FilePath: /nerf-loc/models/nerf_pose_estimator.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
'''
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import traceback
import pycolmap

from .COTR.backbone2d import build_backbone as build_cotr_backbone
from .matcher import Matcher

from .COTR.position_encoding import PositionEmbeddingSine
from nerf_loc.utils.metrics import compute_pose_error

from .utils import camera_project
from nerf_loc.datasets.colmap.read_write_model import qvec2rotmat

from .pose_optimizer import PoseOptimizer
from .conditional_nerf.utils import get_embedder
from .conditional_nerf.model import ConditionalNeRF
from .conditional_nerf.model_simple import ConditionalNeRFSimple
from .appearance_embedding import AppearanceEmbedding, AppearanceAdaptLayer

class NerfPoseEstimator(nn.Module):
    def __init__(self, args, dataset):
        super().__init__()
        self.args = args
        self.dataset = dataset
        self.hidden_dim = hidden_dim = args.matcher_hidden_dim
        if args.backbone2d == 'cotr':
            self.backbone2d = build_cotr_backbone(
                model_path='models/COTR/default/checkpoint.pth.tar', 
                return_layers=['conv1', 'layer1', 'layer2'], 
                train_backbone=True, use_fpn=args.backbone2d_use_fpn, fpn_dim=args.backbone2d_fpn_dim)
        else:
            raise NotImplementedError
        self.backbone2d_coarse_layer_name = args.backbone2d_coarse_layer_name
        self.backbone2d_fine_layer_name = args.backbone2d_fine_layer_name

        if args.encode_appearance:
            print('Use appearance adaptation.')
            self.embedding_a = AppearanceEmbedding(args)
            self.adapt_appearance_coarse = AppearanceAdaptLayer(args, args.backbone2d_fpn_dim)
            self.adapt_appearance_fine = AppearanceAdaptLayer(args, args.backbone2d_fpn_dim)
            if self.args.train_nerf:
                self.adapt_appearance_rgb = AppearanceAdaptLayer(args, 3, is_rgb=True)
        else:
            print('No appearance adaptation.')

        # pc_range = dataset.get_pc_range()

        # adapt parameters for scene scale
        self.coarse_matching_depth_thresh = args.matching.coarse_matching_depth_thresh * dataset.scale_factor
        print('coarse_matching_depth_thresh after scaling: ', self.coarse_matching_depth_thresh)
        ########

        self.proj_layer_2d = \
            nn.Linear(self.backbone2d.layer_to_channels[self.backbone2d_coarse_layer_name], hidden_dim)
        self.pos_emd_3d_fn, _ = get_embedder(hidden_dim//6, 0, include_input=False)
        self.pos_emd_2d_fn = PositionEmbeddingSine(hidden_dim // 2, normalize=True, sine_type='lin_sine')

        self.matcher = Matcher(
            args, 
            hidden_dim, 
            self.backbone2d.layer_to_channels[self.backbone2d_coarse_layer_name], 
            self.backbone2d.layer_to_channels[self.backbone2d_fine_layer_name], 
            fine_matching=True)

        if self.args.cascade_matching:
            self.matcher_fine = Matcher(
                args, 
                hidden_dim, 
                self.backbone2d.layer_to_channels[self.backbone2d_coarse_layer_name], 
                self.backbone2d.layer_to_channels[self.backbone2d_fine_layer_name], 
                fine_matching=True)

        if args.simple_3d_model:
            self.model_3d = ConditionalNeRFSimple(self.args)
        else:
            self.model_3d = ConditionalNeRF(self.args)

        if self.args.optimize_pose:
            self.pose_optimizer = PoseOptimizer(args, self.model_3d, debug=False, use_feat=False)

    def extract_2d(self, imgs):
        """
        Args: 
            imgs: (B,3,H,W)
        Returns: 
            desc: (B,H,W,C)
        """        
        feat_pyramid = self.backbone2d(imgs) # B,C,H,W

        coarse_layer_name = self.backbone2d_coarse_layer_name
        fine_layer_name = self.backbone2d_fine_layer_name

        feat_coarse = feat_pyramid[coarse_layer_name]
        feat_fine = feat_pyramid[fine_layer_name]

        feat_coarse = feat_coarse.permute(0,2,3,1) # B,Hc,Wc,C
        feat_fine = feat_fine.permute(0,2,3,1) # B,Hf,Wf,C

        pos_emd = self.pos_emd_2d_fn(feat_coarse[...,0]) # B,H,W,C

        ret = {
            'feat_rgb': imgs.permute(0,2,3,1), # B,H,W,3
            'feat_pyramid': feat_pyramid,
            'feat_fine': feat_fine,
            'feat_coarse': feat_coarse,
            'pos_emb_coarse': pos_emd,
            'stride_coarse': self.backbone2d.layer_to_stride[coarse_layer_name],
            'stride_fine': self.backbone2d.layer_to_stride[fine_layer_name],
        }

        return ret

    def build_3d_2d_pairs(self, 
            img, pts3d, H, W, K, pose, feat_pyramid_2d=None, thr=0.01, stride=1, depth_map=None, data=None):
        assert ('conv1' in feat_pyramid_2d) and (feat_pyramid_2d['conv1'].shape[0] == 1)

        if depth_map is not None:
            depth = depth_map # H,W
        else:
            depth = None
        # depth = None # force to use nerf depth
        # project 3d points to image and find corresponding 2D points
        with torch.no_grad():
            # Twc = torch.eye(4).to(pose.device)
            # Twc[:3] = pose
            Twc = pose
            pts3d_hom = torch.cat([pts3d, torch.ones_like(pts3d[:,:1])], dim=1)
            pts3d_cam = torch.mm(Twc.inverse(), pts3d_hom.t()).t()
            u, v, z = camera_project(pts3d_cam[:,:3], K)
            proj_valid_mask = (u >= 0) & (v >= 0) & (u < W) & (v < H) & (z > 0)
            uv = torch.stack([u,v], dim=1)
            if self.training and self.args.train_nerf \
                and proj_valid_mask.sum() > 0 and (depth is None or (depth==0).all()):
                # provide depth with nerf when needed
                rays = self.model_3d.points_2d_to_rays(uv[proj_valid_mask], H, W, K, pose)
                rays['depth_range'] = data['depth_range'][0]
                ret = self.model_3d.render_rays(data, rays)
                depth = ret['depth']
            else:
                depth = depth[v[proj_valid_mask].long(), u[proj_valid_mask].long()]
            depth_valid_mask = torch.zeros_like(proj_valid_mask)
            depth_valid_mask[proj_valid_mask] = ((depth - z[proj_valid_mask]).abs() < thr)
            # incorrect depth may leads to few correspondence, we just disable depth check for such case
            pos_mask = proj_valid_mask & depth_valid_mask
            if pos_mask.sum() < 4:
                # print('bad depth map')
                pos_mask = proj_valid_mask

            grid_y, grid_x = torch.meshgrid(torch.arange(H//stride), torch.arange(W//stride))
            pts2d_grid = torch.stack([grid_x, grid_y], dim=-1).view(-1, 2).to(pts3d.device)

            pos_idx_2d = (uv[pos_mask]/stride).long()
            pos_idx_2d = pos_idx_2d[:,0] + pos_idx_2d[:,1] * (W//stride)

            pos_pairs = torch.stack([
                pos_mask.nonzero(as_tuple=True)[0], # 3d idx
                pos_idx_2d, # 2d idx
            ])

            pts3d_proj = uv / stride

        return pts3d, pts2d_grid, pts3d_proj, pos_pairs


    def select_3d_keypoints(self, pts3d, topk_poses, K, H, W):
        # select visible 3D points by topk_poses
        visible_mask = torch.zeros([len(pts3d)], dtype=torch.bool, device=pts3d.device)
        for pose in topk_poses:
            # Twc = torch.eye(4).to(pose.device)
            # Twc[:3] = pose
            pts3d_hom = torch.cat([pts3d, torch.ones_like(pts3d[:,:1])], dim=1)
            pts3d_cam = torch.mm(pose.inverse(), pts3d_hom.t()).t()
            u, v, z = camera_project(pts3d_cam[:,:3], K)
            proj_valid_mask = (u >= 0) & (v >= 0) & (u < W) & (v < H) & (z > 0)
            visible_mask |= proj_valid_mask
        visible_idx = torch.where(visible_mask)[0]
        return visible_idx

    def build_support_set(self, batch):
        topk_images = batch['topk_images'][0]
        topk_depths = batch['topk_depths'][0]
        if 'topk_depths_gt' in batch:
            topk_depths_gt = batch['topk_depths_gt'][0]
        else:
            topk_depths_gt = topk_depths
        topk_poses = batch['topk_poses'][0]
        topk_Ks = batch['topk_Ks'][0]
        if self.training:
            select_view_idx = np.random.choice(
                len(batch['topk_images'][0]), self.args.n_views_train, 
                replace=len(batch['topk_images'][0])<self.args.n_views_train) # randomly select n views
            topk_images = topk_images[select_view_idx]
            topk_depths = topk_depths[select_view_idx]
            topk_depths_gt = topk_depths_gt[select_view_idx]
            topk_poses = topk_poses[select_view_idx]
            topk_Ks = topk_Ks[select_view_idx]
        else:
            topk_images = topk_images[:self.args.n_views_test]
            topk_depths = topk_depths[:self.args.n_views_test]
            topk_depths_gt = topk_depths_gt[:self.args.n_views_test]
            topk_poses = topk_poses[:self.args.n_views_test]
            topk_Ks = topk_Ks[:self.args.n_views_test]
        return topk_images, topk_depths, topk_depths_gt, topk_poses, topk_Ks

    def appearance_adaptation(self, batch, data, outputs, feat_pyramid, feat_pyramid_src):
        if self.args.encode_appearance:
            embedding_a = self.embedding_a(batch['image'], feat_pyramid)
            embedding_a_src = self.embedding_a(data['topk_images'], feat_pyramid_src)
            if self.args.train_nerf:
                adapted_topk_images = self.adapt_appearance_rgb(
                    data['topk_images'].permute(0,2,3,1), embedding_a_src, embedding_a) # B,H,W,3
                data['topk_images'] = adapted_topk_images.permute(0,3,1,2)
                outputs['topk_images_adapted'] = data['topk_images'] # for vis
            data['feat_coarse_src'] = \
                self.adapt_appearance_coarse(data['feat_coarse_src'], embedding_a_src, embedding_a)
            data['feat_fine_src'] = \
                self.adapt_appearance_fine(data['feat_fine_src'], embedding_a_src, embedding_a)
        else:
            embedding_a = None

        data.update({
            'embedding_a': embedding_a
        })
        return data

    def forward(self, batch):
        batch_size = batch['image'].shape[0]
        assert batch_size==1

        pts3d_all = batch['points3d'][...,:3]
        pts3d_rgb_all = batch['points3d'][...,3:6] / 255

        topk_images, topk_depths, topk_depths_gt, topk_poses, topk_Ks = self.build_support_set(batch)
        outputs_2d = self.extract_2d(batch['image'])
        outputs_2d_src = self.extract_2d(topk_images)

        scene = batch['scene'][0]
        pts3d_all = pts3d_all[0]
        pts3d_rgb_all = pts3d_rgb_all[0]
        H,W = batch['image'][0].shape[-2:]

        data = {
            'scene': scene,
            'filename': batch['filename'][0],
            'img': batch['image'][0],
            'depth': batch['depth'][0],
            'K': batch['K'][0],
            'pose': batch['pose'][0],
            'H': H,
            'W': W,
            'pts3d_all': pts3d_all,
            'depth_range': torch.stack([batch['near'], batch['far']], dim=1),
            'topk_images': topk_images,
            'topk_depths': topk_depths,
            'topk_depths_gt': topk_depths_gt,
            'topk_poses': topk_poses,
            'topk_Ks': topk_Ks,
            'feat_coarse_src': outputs_2d_src['feat_coarse'],
            'feat_fine_src': outputs_2d_src['feat_fine'],
        }
        if 'white_bkgd' in batch:
            data['white_bkgd'] = batch['white_bkgd'][0]
        if 'target_mask' in batch:
            data['target_mask'] = batch['target_mask'][0]
        if 'sample_coords' in batch:
            data['sample_coords'] = batch['sample_coords'][0]
        data.update(outputs_2d)

        outputs = {'loss': 0}
        
        data = self.appearance_adaptation(
            batch, data, outputs, 
            outputs_2d['feat_pyramid'], outputs_2d_src['feat_pyramid'])

        # force rebuild support_neural_points
        self.model_3d.support_neural_points = None
        self.model_3d.multiview_aggregator.vis_featmaps = None

        if self.args.train_pose:
            if self.args.keypoints_3d_source == 'sfm':
                if self.training:
                    pose_candidates = torch.cat([data['pose'][None], topk_poses], dim=0)
                else:
                    pose_candidates = topk_poses
                pose_candidates = topk_poses
                
                select_idx = self.select_3d_keypoints(
                    data['pts3d_all'], pose_candidates, data['K'], data['H'], data['W'])
                pts3d = data['pts3d_all'][select_idx]

                target_points = data['pts3d_all'] # use points from external 3d model
                if len(target_points) > self.args.matching.fine_num_3d_keypoints:
                    sample_idx = torch.tensor(
                        np.random.choice(len(target_points), 
                        self.args.matching.fine_num_3d_keypoints, 
                        replace=len(target_points)<self.args.matching.fine_num_3d_keypoints)).long()
                else:
                    sample_idx = torch.arange(len(target_points))
                target_points = target_points[sample_idx]
            else:
                target_points = None
            
            desc_3d, pts3d, pts3d_ndc = self.model_3d.query_coarse(
                data=data,
                points=target_points, 
                # points=None,
                embed_a=data['embedding_a'])

            data.update({
                'pts3d': pts3d,
                'pts3d_ndc': pts3d_ndc,
                'desc_3d': desc_3d
            })
            outputs.update(self.estimate(data, self.matcher, need_pose=False))

            # cascade matching
            if self.args.cascade_matching:
                # select points by visibility
                if self.training:
                    T_init = data['pose'].detach()
                else:
                    T_init = torch.tensor(outputs['T']).float().to(desc_3d.device)
                select_idx = self.select_3d_keypoints(pts3d, [T_init], data['K'], data['H'], data['W'])

                if len(select_idx) > 0:
                    data.update({
                        'pts3d': pts3d[select_idx],
                        'pts3d_ndc': pts3d_ndc[select_idx],
                        'desc_3d': desc_3d[select_idx]
                    })
                    outputs_fine = self.estimate(data, self.matcher_fine, need_pose=False)
                    if self.training:
                        outputs['loss'] += outputs_fine['loss']
                    else:
                        outputs['T'] = outputs_fine['T']
            ###############

        if self.training:
            ref_depth_loss = self.model_3d.multiview_aggregator.compute_ref_depth_loss(
                data['topk_Ks'], data['topk_poses'], data['topk_images'], 
                data['feat_fine_src'].permute(0,3,1,2), 
                data['topk_depths'],
                data['topk_depths_gt'],
                data['depth_range'][0]
            )
            outputs.update({
                'ref_depth_loss': ref_depth_loss
            })
            outputs['loss'] += self.args.ref_depth_loss_weight * ref_depth_loss
        
        if self.training and self.args.train_nerf:
            render_loss, psnr = self.model_3d.compute_render_loss(data)
            outputs.update({
                'render_loss': render_loss,
                'psnr': psnr
            })
            outputs['loss'] += self.args.render_loss_weight * render_loss

        if batch.get('render_image', False):
            n_views = data['topk_Ks'].shape[0]
            Ks_src = torch.eye(4)[None].repeat(n_views,1,1).to(data['pose'].device)
            Ks_src[:,:3,:3] = data['topk_Ks'].clone()
            # Ks_src[:,:2] /= data['stride_fine']
            start = time.time()
            with torch.no_grad():
                ret = self.model_3d.render_image(data)
            end = time.time()
            print(end-start)
            outputs['rendered_image'] = ret['rgb']
            outputs['rendered_depth'] = ret['depth']
            if 'depth_coarse' in ret:
                outputs['rendered_depth_coarse'] = ret['depth_coarse']
            if 'feat' in ret:
                outputs['rendered_feat'] = ret['feat']
                outputs['rendered_feat_gt'] = F.interpolate(
                    data['feat_pyramid']['layer1'], 
                    size=(data['H'], data['W']), mode='bilinear', align_corners=False).permute(0,2,3,1)[0]

        if self.args.optimize_pose:
            pose = data['pose'] # gt
            T_init = torch.tensor(outputs['T']).float().to(data['desc_3d'].device) # predict
            # T_init = data['topk_poses'][0] # nearest pose
            
            rot_err_init, trans_err_init = compute_pose_error(T_init.detach().cpu().numpy(), pose.cpu().numpy())
            print('before pose optimization: ', rot_err_init, trans_err_init)
            T = self.pose_optimizer(T_init, data, max_steps=50, lr=0.001)
            rot_err_refine, trans_err_refine = compute_pose_error(T.detach().cpu().numpy(), pose.cpu().numpy())
            print('after pose optimization: ', rot_err_refine, trans_err_refine)

            T = T.detach().cpu().numpy()
            outputs['T'] = T

        return outputs

    def estimate(self, data, matcher, need_pose=False):
        fine_matching = matcher.fine_matching

        img, pts3d, pose, K, H, W = data['img'], data['pts3d'], data['pose'], data['K'], data['H'], data['W']
        pts3d_ndc = data['pts3d_ndc']
        desc_3d = data['desc_3d']
        scene = data['scene']
        filename = data['filename']

        desc_map = data['feat_coarse']
        pos_emd_map = data['pos_emb_coarse']
        desc_map, pos_emd_map = desc_map[0], pos_emd_map[0]

        # pts3d, desc_3d = self.select_3d_keypoints(pts3d_all, desc_3d_all, [pose], K, H, W)
        pts3d, pts2d, pts3d_proj_gt, pos_pairs = self.build_3d_2d_pairs(
            img, pts3d, H, W, K, pose,
            feat_pyramid_2d=data['feat_pyramid'],
            thr=self.coarse_matching_depth_thresh, 
            stride=data['stride_coarse'],
            depth_map=data['depth'], data=data)

        ### no valid reprojected points on current frame TODO: improve depth map and retrieval
        if self.training and len(pos_pairs[0]) == 0:
            print(f'Error: zero positive pairs {scene} : {filename}')
            sparse_depth = torch.zeros_like(data['depth'])
            w2c = data['pose'].inverse()
            uvz = torch.matmul(data['K'], torch.matmul(w2c[:3,:3], data['pts3d_all'].T) + w2c[:3,3:])
            uv = (uvz[:2] / uvz[2:]).long()
            u = uv[0]
            v = uv[1]
            z = uvz[2]
            mask = (u >= 0) & (u < W) & (v >=0 ) & (v < H) & (z > 0)
            visible_pts3d = data['pts3d_all'][mask]
            sample_idx = torch.tensor(
                np.random.choice(len(visible_pts3d), 
                self.args.matching.fine_num_3d_keypoints, 
                replace=len(visible_pts3d)<self.args.matching.fine_num_3d_keypoints)).long()
            data['desc_3d'], data['pts3d'], data['pts3d_ndc'] = self.model_3d.query_coarse(
                data=data,
                points=visible_pts3d[sample_idx], 
                embed_a=data['embedding_a'])
            desc_3d, pts3d, pts3d_ndc = data['desc_3d'], data['pts3d'], data['pts3d_ndc']
            pts3d, pts2d, pts3d_proj_gt, pos_pairs = self.build_3d_2d_pairs(
                img, pts3d, H, W, K, pose,
                feat_pyramid_2d=data['feat_pyramid'],
                thr=self.coarse_matching_depth_thresh, 
                stride=data['stride_coarse'],
                depth_map=sparse_depth, data=data)
        #####################

        y,x = pts2d[:,1].long(),pts2d[:,0].long()
        desc_2d = self.proj_layer_2d(desc_map[y,x])
        # desc_2d = desc_map[y,x]
        pos_emd_2d = pos_emd_map[y,x]

        # for fine matching
        if fine_matching:
            desc_3d_fine, _, _ = self.model_3d.query_fine(
                data=data,
                points=pts3d, 
                embed_a=data['embedding_a'])

            data.update({'desc_3d_fine': desc_3d_fine})

        pos_emd_3d = self.pos_emd_3d_fn(pts3d_ndc)

        # to fine scale
        pts2d = (pts2d * data['stride_coarse'] / data['stride_fine']).float()
        pts3d_proj_gt = (pts3d_proj_gt * data['stride_coarse'] / data['stride_fine'])

        data.update({
            'kps3d': pts3d,
            'kps2d': pts2d, # in fine scale
            'desc_2d_coarse': desc_2d,
            'pos_emd_3d': pos_emd_3d,
            'pos_emd_2d': pos_emd_2d,
        })
        if self.training:
            # compute conf_matrix target
            conf_matrix_gt = torch.zeros([len(pts3d), len(pts2d)], device=pts2d.device).long()
            conf_matrix_gt[pos_pairs[0], pos_pairs[1]] = 1
            data.update({
                'conf_matrix_gt': conf_matrix_gt,
                'kps3d_proj_gt': pts3d_proj_gt,
                'pairs_gt': pos_pairs,
            })
        match_res = matcher(data)
        if match_res is None:
            # no coarse matching at all
            return {
                'score_matrix': torch.zeros(len(pts3d), len(pts2d)).to(pts3d.device),
                'pairs_gt': pos_pairs,
                'pairs': [torch.tensor([]), torch.tensor([])],
                'mkps2d': [],
                'mkps3d': [],
                'T': np.eye(4)
            }
        pairs = match_res['pairs']
        # to input scale
        mkps3d = match_res['mkps3d']
        if fine_matching:
            mkps2d = match_res['mkps2d_f'] * data['stride_fine']
        else:
            mkps2d = match_res['mkps2d_c'] * data['stride_fine']

        data.update({'pairs': pairs})

        outputs = {
            'score_matrix': match_res['score_matrix'],
            'pairs_gt': pos_pairs,
            'pairs': pairs,
            'mkps2d': mkps2d,
            'mkps3d': mkps3d
        }

        if self.training:
            loss = self.args.coarse_loss_weight * match_res['coarse_loss']
            if fine_matching:
                loss += self.args.fine_loss_weight * match_res['fine_loss']

        if not self.training or need_pose:
            try:
                # TODO: set ransac_thresh according to image resolution
                ransac_thresh = self.args.ransac_thresh
                if not fine_matching:
                    ransac_thresh *= 2
                T, inliers = self.estimate_pose(
                    mkps2d.detach(), mkps3d.detach(), K, W, H, ransac_thresh=ransac_thresh)
            except Exception as err:
                # print('pnp failed')
                traceback.print_exc()
                T = np.eye(4)
            outputs['T'] = T

        if self.training:
            outputs.update({
                'loss': loss,
                # 'desc_3d_fine_loss': desc_3d_fine_loss,
                'coarse_match_loss': self.args.coarse_loss_weight * match_res['coarse_loss'],
                'conf_matrix_gt': conf_matrix_gt
            })
            if fine_matching:
                outputs.update({
                    'fine_match_loss': self.args.fine_loss_weight * match_res['fine_loss'],
                    'fine_err': match_res['fine_err']
                })

        return outputs

    def estimate_pose(self, matched_kps_2d, matched_kps_3d, K, width, height, ransac_thresh=48):
        """
        Args: 
            kps_3d: (N,3)
            kps_2d: (M,3)
        Returns: 
        """      
        fx = K[0,0]
        fy = K[1,1]
        cx = K[0,2]
        cy = K[1,2]
        colmap_camera = {
            'model': 'PINHOLE',
            'width': width,
            'height': height,
            'params': [fx, fy, cx, cy]
        }
        ret = pycolmap.absolute_pose_estimation(
            matched_kps_2d.cpu().numpy(), matched_kps_3d.cpu().numpy(), colmap_camera, ransac_thresh)
        if not ret['success']:
            return None
        R = qvec2rotmat(ret['qvec'])
        t = ret['tvec']
        T = np.eye(4)
        T[:3,:3] = R
        T[:3, 3] = t
        return np.linalg.inv(T), np.array(ret['inliers']) # camera to world
