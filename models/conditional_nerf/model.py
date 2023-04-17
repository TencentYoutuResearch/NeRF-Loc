"""
Author: jenningsliu
Date: 2022-06-30 10:51:58
LastEditors: jenningsliu
LastEditTime: 2022-08-11 12:24:10
FilePath: /nerf-loc/models/pointnerf/pointnerf_3d_model.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from einops import rearrange, reduce, repeat

from models.ops.knn.knn_utils import knn_points, knn_gather
from models.ibrnet.ibrnet import Projector, MultiHeadAttention
from .losses import RenderingLoss
from .ray_unet import RayUnet
from .multiview_aggregator import MultiviewFeatureAggregator
from .utils import get_rays, sample_pdf, get_embedder, img2mse, mse2psnr

class ConditionalNeRF(nn.Module):
    def __init__(self, args, activation_func=nn.LeakyReLU(inplace=True)):
        super().__init__()
        self.args = copy.deepcopy(args)

        self.xyz_embed_fn, xyz_embed_dim = get_embedder(args.multires, args.i_embed)
        self.view_embed_fn, view_embed_dim = get_embedder(args.multires_views, args.i_embed)
        self.ray_diff_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, view_embed_dim),
                                        activation_func)

        support_feature_dim = 3+self.args.backbone2d_fpn_dim # rgb,feature
        W = self.args.model_3d_hidden_dim

        self.multiview_aggregator = MultiviewFeatureAggregator(
            args,
            in_channels=args.backbone2d_fpn_dim, 
            out_channels=W
        )

        # neural points confidence
        self.projector = Projector()
        self.confidence_mlp = nn.Sequential(
            nn.Linear(W, 64),
            activation_func,
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        self.keypoint_head = nn.Sequential(
            nn.Linear(self.args.backbone2d_fpn_dim, 1),
            nn.Sigmoid()
        )

        self.base_mlp = nn.Sequential(
            # nn.Linear(support_feature_dim+3+3+1, W),
            nn.Linear(support_feature_dim+xyz_embed_dim+view_embed_dim, W),
            activation_func,
            nn.Linear(W, W),
            activation_func,
            nn.Linear(W, W),
            activation_func
        )
        self.base_mlp_attn = MultiHeadAttention(4, W, 32, 32)
        self.base_mlp_agg_weight = nn.Sequential(
            nn.Linear(W, W),
            activation_func,
            nn.Linear(W, 1)
        )

        self.support_neural_points = None

        self.ray_unet = RayUnet(W, args.render.N_samples+args.render.N_importance)
        # self.ray_attention = MultiHeadAttention(4, W, W//4, W//4)
        self.sigma_mlp = nn.Sequential(nn.Linear(W, 1), nn.Softplus())
        if self.args.render.render_feature:
            self.feat_mlp = nn.Sequential(
                nn.Linear(W, W),
                activation_func,
                nn.Linear(W, args.backbone2d_fpn_dim)
            )
        self.rgb_blending_mlp = nn.Sequential(
            nn.Linear(W+(3+args.backbone2d_fpn_dim)+1+4, 32),
            activation_func,
            nn.Linear(32, 16),
            activation_func,
            nn.Linear(16, 1)
        )
        # if self.args.use_color_volume:
        #     self.rgb_mlp = nn.Sequential(nn.Linear(W+view_embed_dim+3, 3), nn.Sigmoid())
        # else:
        #     self.rgb_mlp = nn.Sequential(nn.Linear(W+view_embed_dim, 3), nn.Sigmoid())
        # self.rgb_mlp = nn.Sequential(nn.Linear(W+3, 3), nn.Sigmoid())

        # if self.args.encode_appearance:
        #     self.appearance_mlp = nn.Sequential(
        #         nn.Linear(W+self.args.appearance_emb_dim, 64),
        #         activation_func,
        #         nn.Linear(64, 3),
        #         nn.Sigmoid()
        #     )

        if self.args.render.use_render_uncertainty:
            self.beta_mlp = nn.Sequential(nn.Linear(W, 1), nn.Softplus()) # uncertainty
            self.beta_min = 0.1

        if self.args.use_scene_coord_memorization:
            # for per-scene finetuning
            self.coord_desc_mlp_coarse = nn.Sequential(
                nn.Linear(xyz_embed_dim, W),
                nn.ReLU(inplace=True),
                nn.Linear(W, W),
                nn.ReLU(inplace=True),
                nn.Linear(W, self.args.matcher_hidden_dim),
            )
            self.coord_desc_mlp_fine = nn.Sequential(
                nn.Linear(xyz_embed_dim, W),
                nn.ReLU(inplace=True),
                nn.Linear(W, W),
                nn.ReLU(inplace=True),
                nn.Linear(W, self.args.matcher_hidden_dim),
            )

        self.proj_layer_3d_coarse = nn.Linear(W+support_feature_dim, self.args.matcher_hidden_dim)
        self.proj_layer_3d_fine = nn.Linear(W+support_feature_dim, self.args.matcher_hidden_dim)

        self.render_loss = RenderingLoss(use_depth=self.args.use_depth_supervision, depth_std=0.05)

    def estimate_neural_points_confidence(self, 
            points, intrinsics, extrinsics, images, featmaps, depths, depth_range):
        mv_feature, _, _ = \
            self.multiview_aggregator(points, intrinsics, extrinsics, images, featmaps, depths, depth_range)
        conf = self.confidence_mlp(mv_feature)
        return conf

    def build_support_neural_points(self, data):
        topk_depths = data['topk_depths']

        desc_3d_c, pts3d_c, pts3d_ndc_c, dir_c = self.backproject_support_frame(
            data['topk_images'], # V,3,H,W
            data['feat_coarse_src'], # V,H,W,C
            # data['topk_depths'],
            topk_depths,
            data['topk_Ks'],
            data['topk_poses'],
            stride=data['stride_coarse']
        )

        desc_3d_f, pts3d_f, pts3d_ndc_f, dir_f = self.backproject_support_frame(
            data['topk_images'], # V,3,H,W
            data['feat_fine_src'], # V,H,W,C
            # data['topk_depths'],
            topk_depths,
            data['topk_Ks'],
            data['topk_poses'],
            stride=data['stride_fine']
        )

        # conf_c = self.estimate_neural_points_confidence(
        #     pts3d_c, data['topk_Ks'], data['topk_poses'], 
        #     data['topk_images'], data['feat_coarse_src'].permute(0,3,1,2), topk_depths, 
        #     data['depth_range'][0])
        conf_c = torch.ones_like(pts3d_c[:,:1])
        conf_f = self.estimate_neural_points_confidence(
            pts3d_f, data['topk_Ks'], data['topk_poses'], 
            data['topk_images'], data['feat_fine_src'].permute(0,3,1,2), topk_depths, 
            data['depth_range'][0])

        # for sampling
        kp_score_c = self.keypoint_head(desc_3d_c[:,3:])

        self.support_neural_points = {
            'coarse': {
                'xyz': pts3d_c,
                'xyz_ndc': pts3d_ndc_c,
                'feature': desc_3d_c,
                'confidence': conf_c,
                'direction': dir_c, # in world frame
                'keypoint_score': kp_score_c
            },
            'fine': {
                'xyz': pts3d_f,
                'xyz_ndc': pts3d_ndc_f,
                'feature': desc_3d_f,
                'confidence': conf_f,
                'direction': dir_f, # in world frame
                # 'keypoint_score': kp_score_f
            }
        }
        if len(self.support_neural_points['coarse']['xyz']) == 0:
            scene = data['scene']
            filename = data['filename']
            print(f'Error: zero support_neural_points {scene} : {filename}')

    def backproject_support_frame(self, imgs, feats, depths, Ks, c2ws, stride=1):
        """
        Args: 
            imgs: V,3,H,W
            feats: V,h,w,C
            depths: V,H,W
            Ks: V,3,3
            c2ws: V,4,4
        Returns: 
            desc_3ds: (N,3+C) RGB+feature
            pts3d_worlds: (N,3)
            pts3d_refs: (N,3)
        """        
        pts3d_refs = [] # in ref frame
        pts3d_worlds = []
        desc_3ds = []
        direction_distances = []
        w2c_ref = c2ws[0].inverse()
        for img, feat, depth, K, c2w in zip(imgs, feats, depths, Ks, c2ws):
            H = int(img.shape[-2] / stride)
            W = int(img.shape[-1] / stride) # feature map size
            K = K.clone()
            K[:2] /= stride
            depth = F.interpolate(depth[None,None], size=(H,W)).squeeze()
            img = F.interpolate(img[None], size=(H,W)).squeeze().permute(1,2,0)
            # H,W = img.shape[-2:]
            # img = img.permute(1,2,0)
            v, u = torch.nonzero(depth > 0, as_tuple=True)
            z = depth[v,u]
            uv_hom = torch.stack([u,v,torch.ones_like(u)], dim=0).float() # 3,N
            pts3d_cam = torch.matmul(K.inverse(), uv_hom) * z
            pts3d_cam_hom = torch.cat([pts3d_cam, torch.ones_like(pts3d_cam[:1])])
            pts3d_world = torch.matmul(c2w[:3,:3], pts3d_cam) + c2w[:3,3:]

            src2ref = torch.matmul(w2c_ref, c2w)
            pts3d_ref = torch.matmul(src2ref, pts3d_cam_hom)[:3]

            rays_o, rays_d = get_rays(H, W, K, c2w)
            direction_distance = torch.cat([
                rays_d[v,u], # viewing direction
                z.view(-1,1) # viewing distance
            ], dim=1)

            pixel_locations = torch.stack([u,v], dim=1) # N,2
            resize_factor = torch.tensor([W-1., H-1.]).to(pixel_locations.device)[None, :]
            normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.
            N = len(pixel_locations)

            desc_3d = torch.cat([
                img[v,u], # rgb
                feat[v,u], # feature
                # torch.cat(feat_sampled_list, dim=1)
            ], dim=1)
            pts3d_refs.append(pts3d_ref.T)
            pts3d_worlds.append(pts3d_world.T)
            desc_3ds.append(desc_3d)
            direction_distances.append(direction_distance)
        pts3d_refs = torch.cat(pts3d_refs)
        pts3d_worlds = torch.cat(pts3d_worlds)
        desc_3ds = torch.cat(desc_3ds)
        direction_distances = torch.cat(direction_distances)

        return desc_3ds, pts3d_worlds, pts3d_refs, direction_distances

    def sample_points_3d(self):
        n_points = len(self.support_neural_points['coarse']['xyz'])
        sample_idx = torch.multinomial(
            self.support_neural_points['coarse']['keypoint_score'].squeeze(1), 
            self.args.matching.fine_num_3d_keypoints, 
            replacement=n_points<self.args.matching.fine_num_3d_keypoints)
        pts3d = self.support_neural_points['coarse']['xyz'][sample_idx]
        pts3d_ndc = self.support_neural_points['coarse']['xyz_ndc'][sample_idx]
        return pts3d, pts3d_ndc, sample_idx

    def query_coarse(self, data, points=None, embed_a=None):        
        if self.support_neural_points is None:
            self.build_support_neural_points(data)

        if points is None:
            # position from fine points, feature from coarse points
            pts3d, pts3d_ndc, sample_idx = self.sample_points_3d()
            feature_2d = self.support_neural_points['coarse']['feature'][sample_idx]
        else:
            pts3d = points
            w2c_ref = data['topk_poses'][0].inverse()
            pts3d_ndc = (torch.matmul(w2c_ref[:3,:3], points.T) + w2c_ref[:3,3:]).T
            _, idx, _ = knn_points(
                points.unsqueeze(0), 
                self.support_neural_points['coarse']['xyz'].unsqueeze(0), 
                K=1, return_nn=True) # 1,N,K,
            feature_2d = knn_gather(self.support_neural_points['coarse']['feature'][None], idx)[0] # N,K,C
            feature_2d = feature_2d.squeeze(1)

        query_dict = self.query(
            data, pts3d, 
            support_featmaps=data['feat_coarse_src'].permute(0,3,1,2),
            support_neural_points=self.support_neural_points['coarse'],
            K=8,
            embed_a=embed_a
        )
        desc_3d = self.proj_layer_3d_coarse(torch.cat([
            query_dict['feature_agg'], 
            feature_2d
        ], dim=1))

        if self.args.use_scene_coord_memorization:
            coord_desc_3d = self.coord_desc_mlp_coarse(self.xyz_embed_fn(pts3d))
            desc_3d += coord_desc_3d

        return desc_3d, pts3d, pts3d_ndc

    def query_fine(self, data, points, embed_a=None):
        if self.support_neural_points is None:
            self.build_support_neural_points(data)

        _, idx, _ = knn_points(
            points.unsqueeze(0), 
            self.support_neural_points['fine']['xyz'].unsqueeze(0), 
            K=1, return_nn=True) # 1,N,K,
        feature_2d = knn_gather(self.support_neural_points['fine']['feature'][None], idx)[0] # N,K,C
        feature_2d = feature_2d.squeeze(1)

        query_dict = self.query(
            data, points, 
            support_featmaps=data['feat_fine_src'].permute(0,3,1,2),
            support_neural_points=self.support_neural_points['fine'],
            K=1,
            embed_a=embed_a
        )

        desc_3d = self.proj_layer_3d_fine(torch.cat([
            query_dict['feature_agg'], 
            feature_2d
        ], dim=1))

        if self.args.use_scene_coord_memorization:
            coord_desc_3d = self.coord_desc_mlp_fine(self.xyz_embed_fn(points))
            desc_3d += coord_desc_3d

        return desc_3d, None, None
    
    def query(self, 
            data, xyz, support_featmaps, support_neural_points, 
            direction=None, K=8, embed_a=None, target_proj_mat=None):
        """
        Args: 
            data:
            xyz: target points in world frame (N,3)
            support_featmaps: feature maps of support views (V,C,h,w)
            support_neural_points:
                xyz: (M,3)
                feature: (M,C)
                confidence: (M,1)
            direction: (N,4) viewing direction and distance of sample points. in world frame
            embed_a: (1,C_a)
            target_proj_mat: (3,4)
        Returns: 
            feature: K-nearest support features (N,K,C1)
            weights: K-nearest support weights (N,K)
            feature_agg: aggregated support feature (N,C1)
        """        
        multiview_feature_agg, multiview_feature, multiview_visibility = self.multiview_aggregator(
            xyz,
            data['topk_Ks'], data['topk_poses'], data['topk_images'], 
            support_featmaps,
            data['topk_depths'],
            data['depth_range'][0]) # N,1,C

        # knn search and run mlp
        support_xyz = support_neural_points['xyz'] # M,3
        support_feature = support_neural_points['feature'] # M,C
        support_confidence = support_neural_points['confidence'] # M,1
        support_direction = support_neural_points['direction'] # M,4
        dists, idx, neighbors_xyz = \
            knn_points(xyz.unsqueeze(0), support_xyz.unsqueeze(0), K=K, return_nn=True) # 1,N,K,
        dists = dists[0].sqrt()

        neighbors_xyz = neighbors_xyz[0] # N,K,3
        neighbors_feature = knn_gather(support_feature[None], idx)[0] # N,K,C
        neighbors_confidence = knn_gather(support_confidence[None], idx)[0] # N,K,1
        neighbors_direction = knn_gather(support_direction[None], idx)[0] # N,K,4
        neighbors = {
            'xyz': neighbors_xyz,
            'feature': neighbors_feature,
            'confidence': neighbors_confidence,
            'direction': neighbors_direction
        }

        if direction is None:
            direction = neighbors_direction[:,0,:]

        xyz_offset = xyz[:,None,:].repeat(1,K,1)-neighbors_xyz
        # compute angle differences
        ray_diff = direction[:,:3].unsqueeze(1) - neighbors['direction'][...,:3] # N,8,3
        ray_diff = ray_diff / (torch.norm(ray_diff, dim=-1, keepdim=True) + 1e-8)
        ray_diff_dot = torch.sum(direction[:,:3].unsqueeze(1) * neighbors['direction'][...,:3], dim=-1, keepdim=True)
        ray_diff = torch.cat([ray_diff, ray_diff_dot], dim=-1) # N,8,4
        # # compute distance differences
        # dist_diff = direction[:,3:].unsqueeze(1) - neighbors['direction'][...,3:] # N,8,1

        near, far = data['depth_range'][0]
        point_feature = self.base_mlp(torch.cat([
            neighbors_feature,
            self.xyz_embed_fn(xyz_offset / (far - near)),
            self.ray_diff_fc(ray_diff),
            # dist_diff
        ], dim=-1)) # N,K,C1

        # local_mask = dists < 0.02 * (far - near) # truncat too far away points [N,K]

        feature, _ = \
            self.base_mlp_attn(multiview_feature_agg.unsqueeze(1).repeat(1,K,1), point_feature, point_feature)
        correlation = F.softmax(self.base_mlp_agg_weight(feature).squeeze(-1), dim=1) # N,K

        # sigma = self.sigma_mlp(feature) # N,K,1

        dist_weights = weights = 1. / torch.clamp(dists, min=1e-8) # N,K
        
        # weights += correlation
        # weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-8)
        # TODO: when weights is one-hot, correlation has no effect
        weights *= correlation
        weights *= neighbors_confidence.squeeze(-1)
        weights = weights / torch.clamp(weights.sum(dim=1, keepdim=True), min=1e-8) # N,K
        feature_agg = (feature * weights.unsqueeze(-1)).sum(dim=1) # N,C1

        # return feature_agg, multiview_feature, multiview_visibility
        return {
            'feature_agg': feature_agg,
            'feature': feature,
            'weights': weights,
            'multiview_feature': multiview_feature,
            'multiview_visibility': multiview_visibility
        }

    def query_rgb(self, data, xyz):
        # fetch rgb directly from support images
        intrinsics = data['topk_Ks']
        extrinsics = data['topk_poses']
        images = data['topk_images']
        featmaps = data['feat_fine_src'].permute(0,3,1,2)
        intrinsics_hom = torch.eye(4, device=intrinsics.device).expand(intrinsics.shape[0], 4, 4).clone()
        intrinsics_hom[:,:3,:3] = intrinsics
        rgb, feat, mask = self.projector.compute(xyz, intrinsics_hom, extrinsics, images, featmaps)
        weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        rgb_avg = (rgb * weight).sum(dim=1)
        return rgb_avg

    def sample_depths(self, N_samples, near, far):
        # Sample depth points
        z_steps = torch.linspace(0, 1, N_samples, device=near.device)
        if not self.args.render.lindisp: # use linear sampling in depth space
            z_vals = near * (1-z_steps) + far * z_steps
        else: # use linear sampling in disparity space
            z_vals = 1/(1/near * (1-z_steps) + 1/far * z_steps)
        return z_vals

    def ray_pos_encoding(self, d_model, length, device):
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                            -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        return pe.to(device).unsqueeze(0)

    def render_rays(self, data, rays):
        if self.support_neural_points is None:
            self.build_support_neural_points(data)
        near, far = rays['depth_range']
        N_samples = self.args.render.N_samples

        rays_o, rays_d = rays['rays_o'], rays['rays_d']
        N_rays = rays_o.shape[0]
        rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
        rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

        z_vals = self.sample_depths(N_samples, near, far)
        z_vals = z_vals.expand(N_rays, N_samples).contiguous()

        depth_coarse = None
        if self.args.render.N_importance > 0:
            # infer alpha values of target view from reference views to guide sampling
            z_vals_coarse = self.sample_depths(64, near, far).expand(N_rays, 64).contiguous()
            weights_coarse = self.multiview_aggregator.predict_weights_from_neuray(data, rays, z_vals_coarse)
            depth_coarse = reduce(weights_coarse*z_vals_coarse, 'n1 n2 -> n1', 'sum')
            z_vals_mid = \
                0.5 * (z_vals_coarse[: ,:-1] + z_vals_coarse[: ,1:]) # (N_rays, N_samples-1) interval mid points
            z_vals_fine = sample_pdf(z_vals_mid, weights_coarse[:, 1:-1].detach(), self.args.render.N_importance)
            z_vals = torch.sort(torch.cat([z_vals, z_vals_fine], -1), -1)[0]
            N_samples = self.args.render.N_samples + self.args.render.N_importance

        xyz = rays_o + rays_d * rearrange(z_vals, 'n1 n2 -> n1 n2 1')

        xyz_flat = xyz.view(-1, 3) # N,3
        dir_d_flat = torch.cat([
            rays_d.repeat(1,N_samples,1).view(-1,3),
            z_vals.view(-1,1)
        ], dim=-1) # N,4

        K_h = torch.eye(4).to(data['K'].device)
        K_h[:3,:3] = data['K']
        projection_mat = torch.matmul(K_h, data['pose'].inverse())
        query_dict = self.query(
            data, xyz_flat, 
            support_featmaps=data['feat_fine_src'].permute(0,3,1,2),
            direction=dir_d_flat,
            support_neural_points=self.support_neural_points['fine'],
            K=8,
            embed_a=data['embedding_a'],
            target_proj_mat=projection_mat[:3]
        )
        feature_agg, multiview_feature, multiview_visibility = \
            query_dict['feature_agg'], query_dict['multiview_feature'], query_dict['multiview_visibility']

        # ray_unet
        geometry_feature_agg = self.ray_unet(
            feature_agg.view(N_rays, N_samples, -1).permute(0,2,1)).permute(0,2,1).reshape(N_rays*N_samples, -1)
        
        sigma = self.sigma_mlp(geometry_feature_agg) # N,1

        # color blending
        num_views = multiview_feature.shape[1]
        rgb_in = multiview_feature[:,:,:3]
        ray_diff = self.projector.compute_angle(xyz_flat, data['pose'], data['topk_poses']) # V,N,4
        ray_diff = ray_diff.permute(1,0,2) # N,V,4
        x_rgb = torch.cat([
            feature_agg.unsqueeze(1).expand(-1, num_views, -1), 
            multiview_feature, multiview_visibility, ray_diff], dim=-1)
        blending_weights = self.rgb_blending_mlp(x_rgb)
        blending_weights = blending_weights.masked_fill(multiview_visibility == 0, -1e9)
        blending_weights = F.softmax(blending_weights, dim=1)  # color blending
        rgb = rgb_blended = torch.sum(rgb_in*blending_weights, dim=1) # N,3

        # rendering
        sigma = rearrange(sigma, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
        rgb = rearrange(rgb, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
        # Convert these values using volume rendering
        deltas = z_vals[:, 1:] - z_vals[:, :-1] # (N_rays, N_samples_-1)
        delta_inf = 1e2 * torch.ones_like(deltas[:, :1]) # (N_rays, 1) the last delta is infinity
        deltas = torch.cat([deltas, delta_inf], -1)  # (N_rays, N_samples_)

        alphas = 1-torch.exp(-deltas*sigma.squeeze(-1))
        alphas_shifted = \
            torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1) # [1, 1-a1, 1-a2, ...]
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1) # [1, 1-a1, (1-a1)(1-a2), ...]

        weights = alphas * transmittance
        weights_sum = reduce(weights, 'n1 n2 -> n1', 'sum')

        rgb = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*rgb, 'n1 n2 c -> n1 c', 'sum')
        if data.get('white_bkgd', self.args.render.white_bkgd):
            rgb = rgb + (1-rearrange(weights_sum, 'n -> n 1'))
        depth = reduce(weights*z_vals, 'n1 n2 -> n1', 'sum')
        depth_uncertainty = reduce(weights*((z_vals-depth[:,None])**2), 'n1 n2 -> n1', 'sum')

        # compute valid mask
        intrinsics = data['topk_Ks']
        extrinsics = data['topk_poses']
        intrinsics_hom = torch.eye(4, device=intrinsics.device).expand(intrinsics.shape[0], 4, 4).clone()
        intrinsics_hom[:,:3,:3] = intrinsics
        pixel_locations, project_depths, mask_in_front = \
            self.projector.compute_projections(xyz_flat, intrinsics_hom, extrinsics) # V,N,?
        h,w = data['topk_images'].shape[-2:]
        inbound = self.projector.inbound(pixel_locations, h, w)
        valid_mask = (inbound * mask_in_front).float().permute(1,0) # [N, V]
        # [N_rays, N_samples], should at least have 2 observations
        valid_mask = valid_mask.view(N_rays, N_samples, -1).sum(dim=2) > 1
        # should at least have 8 valid observation on the ray, otherwise don't consider its loss
        valid_mask = valid_mask.float().sum(dim=1) > 8 

        outputs = {
            'rgb': rgb,
            'depth': depth,
            'weights': weights,
            'mask': valid_mask,
            'depth_uncertainty': depth_uncertainty
        }
        if depth_coarse is not None:
            outputs['depth_coarse'] = depth_coarse

        if self.training and self.args.render.use_render_uncertainty:
            beta = self.beta_mlp(geometry_feature_agg)  # N,8,1
            beta = rearrange(beta, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
            beta = reduce(weights*beta.squeeze(-1), 'n1 n2 -> n1', 'sum')
            beta += self.beta_min
            outputs['beta'] = beta

        if self.args.render.render_feature:
            feat = self.feat_mlp(feature_agg)
            feat = rearrange(feat, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples)
            feat = reduce(rearrange(weights, 'n1 n2 -> n1 n2 1')*feat, 'n1 n2 c -> n1 c', 'sum')
            outputs['feat'] = feat

        return outputs

    def render_image(self, data):
        H,W,K,pose = data['H'], data['W'], data['K'], data['pose']
        rays_o, rays_d = get_rays(H, W, K, pose)
        rays_o = rays_o.view(-1,3)
        rays_d = rays_d.view(-1,3)

        u, v = torch.meshgrid(
            torch.linspace(0, W-1, W), torch.linspace(0, H-1, H)
        ) # pytorch's meshgrid has indexing='ij'
        u = u.t().to(K.device).view(-1)
        v = v.t().to(K.device).view(-1)
        pixel_coordinates = torch.stack([u,v], dim=1)

        all_ret = {}
        chunk = self.args.render.chunk
        for i in range(0, rays_o.shape[0], chunk):
            # ret = render_rays(rays_flat[i:i+chunk], **kwargs)
            ray_batch = {
                'pixel_coordinates': pixel_coordinates[i:i+chunk],
                'K': K,
                'pose': pose,
                'H': H,
                'W': W,
                'rays_o': rays_o[i:i+chunk],
                'rays_d': rays_d[i:i+chunk],
                'depth_range': data['depth_range'][0]
            }
            ret = self.render_rays(data, ray_batch)
            for k in ret:
                if k not in all_ret:
                    all_ret[k] = []
                all_ret[k].append(ret[k])
        for k in all_ret:
            all_ret[k] = torch.cat(all_ret[k], 0).view(H,W,-1)
        # mask_out non-care area
        if 'target_mask' in data:
            all_ret['rgb'] *= data['target_mask'][:,:,None].float()
        return all_ret

    def compute_render_loss(self, data):
        if 'sample_coords' in data:
            rays = self.points_2d_to_rays(data['sample_coords'], data['H'], data['W'], data['K'], data['pose'])
        else:
            rays = self.sample_rays(
                self.args.render.N_rand, data['H'], data['W'], 
                data['K'], data['pose'], data.get('target_mask', None)
            )
        rays_o, rays_d = rays['rays_o'], rays['rays_d']
        uv = rays['pixel_coordinates'].long()
        rgb_target = data['img'].permute(1,2,0)[uv[:,1], uv[:,0]]

        targets = {
            'rgb': rgb_target
        }

        rays['depth_range'] = data['depth_range'][0]
        preds = self.render_rays(data, rays)
        mask = preds['mask']

        if self.args.use_depth_supervision:
            depth_target = data['depth'][uv[:,1], uv[:,0]]
            targets.update({
                'depth_range': data['depth_range'][0],
                'depth': depth_target
            })

        if self.args.render.render_feature:
            target_feat_map = F.interpolate(
                data['feat_pyramid']['layer1'], 
                size=(data['H'], data['W']), mode='bilinear', align_corners=False).permute(0,2,3,1)
            feat_target = target_feat_map[0, uv[:,1], uv[:,0]]
            targets.update({
                'feat': feat_target
            })
        if 'target_mask' in data:
            mask = mask & data['target_mask'][uv[:,1], uv[:,0]]
            targets.update({
                'mask': mask
            })

        loss = self.render_loss(preds, targets)

        psnr = mse2psnr(img2mse(preds['rgb'], targets['rgb'], mask=mask))
        return loss, psnr

    def points_2d_to_rays(self, pts2d, H, W, K, pose):
        x, y = pts2d[:,0].long(), pts2d[:,1].long()
        rays_o, rays_d = get_rays(H,W,K,pose)
        # rays_o = rays_o[y,x]
        # rays_d = rays_d[y,x]
        return {
            'pose': pose,
            'K': K,
            'H': H,
            'W': W,
            'pixel_coordinates': pts2d,
            'rays_o': rays_o[y,x],
            'rays_d': rays_d[y,x]
        }

    def sample_rays(self, n_rays, H, W, K, pose, mask=None):
        u, v = torch.meshgrid(torch.arange(W), torch.arange(H))
        u = u.reshape(-1).float()
        v = v.reshape(-1).float()
        pts2d = torch.stack([u,v], dim=1)
        if mask is not None:
            pts2d = pts2d[mask[v.long(),u.long()].bool()]
        idx = np.random.choice(len(pts2d), n_rays, replace=False)

        rays = self.points_2d_to_rays(pts2d[idx], H, W, K, pose)

        return rays
