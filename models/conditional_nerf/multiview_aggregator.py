"""
Author: jenningsliu
Date: 2022-07-18 16:00:37
LastEditors: jenningsliu
LastEditTime: 2022-08-10 19:07:20
FilePath: /nerf-loc/models/pointnerf/multiview_aggregator.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ibrnet.ibrnet import fused_mean_variance, Projector, MultiHeadAttention
from .depth_fusion import DepthFusionNet, project_points_dict, interpolate_feature_map, depth2points, depth2inv_dists
from .visibility_decoder import MixtureLogisticsDistDecoder
from .losses import to_inverse_normalized_depth

class MultiviewFeatureAggregator(nn.Module):
    def __init__(self, args, in_channels, out_channels, hidden_dim=64, activation_func=nn.ELU(inplace=True)):
        super().__init__()
        self.args = args
        self.projector = Projector()

        self.alpha_value_ground_state = -15
        self.depth_fusion = DepthFusionNet(in_channels=in_channels)
        self.vis_featmaps = None
        self.dist_decoder = MixtureLogisticsDistDecoder({})

        self.out_fc = nn.Sequential(
            nn.Linear((in_channels+3)*2+2+1, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, out_channels),
            activation_func,
        )

    def predict_ref_depths(self, intrinsics, extrinsics, images, featmaps, depths, depth_range):
        if self.vis_featmaps is None:
            self.vis_featmaps = self.depth_fusion(images, featmaps, depths, intrinsics, extrinsics, depth_range)
        depth_range = depth_range.view(1,2).repeat(images.shape[0], 1).float()
        V,C,h,w = self.vis_featmaps.shape
        ref_feats = self.vis_featmaps.view(V,C,-1).permute(0,2,1) # V,N,C
        mean = self.dist_decoder.predict_mean(ref_feats)
        ref_depths_pred = \
            self.dist_decoder.decode_ref_depths(mean, depth_range) # [V,N] predicted depth of reference views
        return ref_depths_pred.view(V,h,w)

    def compute_ref_depth_loss(self, intrinsics, extrinsics, images, featmaps, depths, depths_gt, depth_range):
        near, far = depth_range
        ref_depths_pred = self.predict_ref_depths(intrinsics, extrinsics, images, featmaps, depths, depth_range)
        V,h,w = ref_depths_pred.shape
        depths_gt = F.interpolate(depths_gt.unsqueeze(1), size=(h,w)).view(V,-1)
        mask = depths_gt > 0
        ref_depths_pred = ref_depths_pred.view(V, -1)
        depths_gt_norm = to_inverse_normalized_depth(depths_gt, near, far)
        ref_depths_pred_norm = to_inverse_normalized_depth(ref_depths_pred, near, far)
        loss = ((depths_gt_norm - ref_depths_pred_norm)[mask] ** 2).mean() # l2

        return loss

    def predict_visibility(self, ref_imgs_info, que_pts):        
        prj_dict = project_points_dict(ref_imgs_info, que_pts) # V,N,C
        """
        prj_dict 
           depth: projected depth of que_pts to each ref views [V,N,1]
           mask: [V,N,1]
           ray_feats: [V,N,C]
           rgb: [V,N,3]
           dir: [V,N,3]
           pts: [V,N,2]
        """
        
        V, N, _ = prj_dict['mask'].shape
        depth_range = ref_imgs_info['depth_range']
        # decode ray prob
        prj_mean, prj_var, prj_vis, prj_aw = self.dist_decoder(prj_dict['ray_feats'])

        ref_depths = \
            self.dist_decoder.decode_ref_depths(prj_mean, depth_range) # [V,N] predicted depth of reference views

        depth_diff = \
            (prj_dict['depth'].squeeze(-1) - ref_depths).abs() / (depth_range[:,1:]-depth_range[:,:1]) # [V,N]

        visibility = self.dist_decoder.compute_visibility(
            prj_dict['depth'], 
            prj_mean, prj_var, prj_vis, prj_aw,
            depth_range
        )
        # post process
        visibility = visibility.reshape(V,N,1) * prj_dict['mask']
        return visibility, depth_diff

    def predict_weights_from_neuray(self, data, rays, que_depth):
        ref_Ks, ref_poses, ref_images, ref_feats, ref_depths = \
            data['topk_Ks'], data['topk_poses'], data['topk_images'], \
            data['feat_fine_src'].permute(0,3,1,2), data['topk_depths']
        depth_range = data['depth_range'][0]
        if self.vis_featmaps is None:
            self.vis_featmaps = self.depth_fusion(ref_images, ref_feats, ref_depths, ref_Ks, ref_poses, depth_range)
        
        ref_imgs_info = {
            'depth': ref_depths.unsqueeze(1),
            'imgs': ref_images,
            'poses': ref_poses.inverse()[:,:3], # w2c [V,3,4]
            'Ks': ref_Ks,
            'depth_range': depth_range.view(1,2).repeat(ref_images.shape[0], 1).float(),
            'ray_feats': self.vis_featmaps
        }

        que_imgs_info = {
            'coords': rays['pixel_coordinates'][None].to(ref_Ks.device),
            'Ks': rays['K'][None],
            'poses': rays['pose'][None].inverse()[:,:3],
            'depth_range': depth_range[None].float()
        }

        rn,dn = que_depth.shape
        rfn = len(ref_images)

        que_depth = que_depth[None] # 1,rn,dn
        que_dists = depth2inv_dists(que_depth, que_imgs_info['depth_range']) # 1,rn,dn
        que_pts, que_dir = depth2points(que_imgs_info, que_depth) # 1,rn,dn,3

        prj_dict = project_points_dict(ref_imgs_info, que_pts.view(-1,3))
        prj_mean, prj_var, prj_vis, prj_aw = self.dist_decoder(prj_dict['ray_feats'])
        alpha_values, visibility, _ = self.dist_decoder.compute_prob(
            prj_dict['depth'].view(rfn,1,rn,dn), 
            que_dists.view(1,1,rn,dn), 
            prj_mean.view(rfn,1,rn,dn,-1), 
            prj_var.view(rfn,1,rn,dn,-1), 
            prj_vis.view(rfn,1,rn,dn,-1), 
            prj_aw.view(rfn,1,rn,dn,-1), 
            True, ref_imgs_info['depth_range'])

        # post process
        mask = prj_dict['mask'].view(rfn,1,rn,dn,1)
        alphas = alpha_values.reshape(rfn,1,rn,dn,1) * mask + \
                            (1 - mask) * self.alpha_value_ground_state
        vis = visibility.reshape(rfn,1,rn,dn,1) * mask

        alphas = (alphas * vis).sum(0) / torch.clip(vis.sum(0), min=1e-8) # qn,rn,dn,1
        invalid_ray_mask = torch.sum(mask.int().squeeze(-1), 0) == 0
        alphas = alphas * (1 - invalid_ray_mask.float().unsqueeze(-1)) + \
                invalid_ray_mask.float().unsqueeze(-1) * self.alpha_value_ground_state

        alphas = self.dist_decoder.decode_alpha_value(alphas).squeeze(0).squeeze(-1) # rn,dn

        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]
        weights = alphas * transmittance
        # hit_prob = alpha_values2hit_prob(alphas)
        return weights

    def forward(self, 
        sampled_points,
        intrinsics, extrinsics, images, featmaps, depths, depth_range):
        """
        Args: 
            points: N,3
            intrinsics: V,3,3
            extrinsics: V,4,4
            images: V,3,H,W
            featmaps: V,C,H,W
            depths: V,H,W
            depth_range: near far
        Returns: 
            feat_agg: N,C
        """
        intrinsics_hom = torch.eye(4, device=intrinsics.device).expand(intrinsics.shape[0], 4, 4).clone()
        intrinsics_hom[:,:3,:3] = intrinsics
        rgb, feat, mask = self.projector.compute(sampled_points, intrinsics_hom, extrinsics, images, featmaps)
        rgb_feat = torch.cat([rgb, feat], dim=-1) # raw multiview projection [N,V,3+C]
        num_views = rgb_feat.shape[1]

        ### 
        if self.vis_featmaps is None:
            self.vis_featmaps = self.depth_fusion(images, featmaps, depths, intrinsics, extrinsics, depth_range)

        ref_imgs_info = {
            'depth': depths.unsqueeze(1),
            'imgs': images,
            'poses': extrinsics.inverse()[:,:3], # w2c [V,3,4]
            'Ks': intrinsics,
            'depth_range': depth_range.view(1,2).repeat(images.shape[0], 1).float(),
            'ray_feats': self.vis_featmaps
        }

        N_samples = self.args.render.N_samples
        # sampled_dists = depth2inv_dists(
        #     sampled_depths.view(1, -1, N_samples), 
        #     depth_range.view(1,2).float()
        # )
        vis, depth_diff = self.predict_visibility(
            ref_imgs_info,
            sampled_points,
        )
        vis = vis.view(num_views, -1, 1).permute(1,0,2) # N,V,1
        depth_diff = depth_diff.view(num_views, -1, 1).permute(1,0,2) # N,V,1

        weight = vis / (torch.sum(vis, dim=1, keepdim=True) + 1e-8) # [N,V,1]

        # recompute mean and variance considering occlusion
        rgb_feat_mean, rgb_feat_var = fused_mean_variance(rgb_feat, weight) # [N, 1, 3+C]
        depth_diff_mean, depth_diff_var = fused_mean_variance(depth_diff, weight) # [N, 1, 1]
        globalfeat = torch.cat([
            rgb_feat_mean, 
            rgb_feat_var, 
            depth_diff_mean, 
            depth_diff_var
        ], dim=-1)  # [N, 1, 2*(3+C)+2]
        feat_agg = torch.cat([globalfeat.squeeze(1), weight.mean(dim=1)], dim=-1)  # [N, C*2+1]

        # views_feat = torch.cat([
        #     vis,
        #     globalfeat.expand(-1, num_views, -1), 
        #     rgb_feat
        # ], dim=-1)  # [N, V, 1+3*(3+C)]

        out = self.out_fc(feat_agg)
        return out, rgb_feat, vis
