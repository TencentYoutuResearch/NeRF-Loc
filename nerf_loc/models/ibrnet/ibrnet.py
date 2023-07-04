import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
torch._C._jit_set_profiling_executor(False)
torch._C._jit_set_profiling_mode(False)

@torch.jit.script
def fused_mean_variance(x, weight):
    mean = torch.sum(x*weight, dim=1, keepdim=True)
    var = torch.sum(weight * (x - mean)**2, dim=1, keepdim=True)
    return mean, var


# default tensorflow initialization of linear layers
def weights_init(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.zeros_(m.bias.data)


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        # self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
            # attn = attn * mask

        attn = F.softmax(attn, dim=-1)
        # attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(F.relu(self.w_1(x)))
        # x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        # self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        # q = self.dropout(self.fc(q))
        q = self.fc(q)
        q += residual

        q = self.layer_norm(q)

        return q, attn


class Projector():
    def __init__(self):
        pass

    def inbound(self, pixel_locations, h, w):
        '''
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        '''
        return (pixel_locations[..., 0] <= w - 1.) & \
               (pixel_locations[..., 0] >= 0) & \
               (pixel_locations[..., 1] <= h - 1.) &\
               (pixel_locations[..., 1] >= 0)

    def normalize(self, pixel_locations, h, w):
        resize_factor = torch.tensor([w-1., h-1.]).to(pixel_locations.device)[None, None, :]
        normalized_pixel_locations = 2 * pixel_locations / resize_factor - 1.  # [n_views, n_points, 2]
        return normalized_pixel_locations

    def compute_angle(self, xyz, query_pose, train_poses):
        '''
        :param xyz: [N, 3]
        :param query_pose: [4,4]
        :param train_poses: [n_views, 4, 4]
        :return: [n_views, N, 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        '''
        # original_shape = xyz.shape[:2]
        # xyz = xyz.reshape(-1, 3)
        # train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]
        num_views = len(train_poses)
        query_pose = query_pose.reshape(1, 4, 4).repeat(num_views, 1, 1)  # [n_views, 4, 4]
        ray2tar_pose = (query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2tar_pose_norm = ray2tar_pose / (torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6)
        ray2train_pose = (train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(0))
        ray2train_pose_norm = ray2train_pose / (torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6)
        ray_diff = ray2tar_pose_norm - ray2train_pose_norm
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(ray2tar_pose_norm * ray2train_pose_norm, dim=-1, keepdim=True)
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat([ray_diff_direction, ray_diff_dot], dim=-1)
        ray_diff = ray_diff.reshape(num_views, -1, 4)
        return ray_diff

    def compute_projections(self, xyz, intrinsics, poses):
        '''
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [n_views, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        '''
        original_shape = xyz.shape[:2]
        xyz = xyz.reshape(-1, 3)
        num_views = len(intrinsics)
        # train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [n_views, 4, 4]
        # train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [n_views, 4, 4]


        xyz_h = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)  # [n_points, 4]
        projections = intrinsics.bmm(torch.inverse(poses)) \
            .bmm(xyz_h.t()[None, ...].repeat(num_views, 1, 1))  # [n_views, 4, n_points]
        projections = projections.permute(0, 2, 1)  # [n_views, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(projections[..., 2:3], min=1e-8)  # [n_views, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        depths = projections[..., 2]
        mask = projections[..., 2] > 0   # a point is invalid if behind the camera
        # from IPython import embed;embed()
        return pixel_locations, depths, mask

    def compute(self, xyz, intrinsics, extrinsics, images, featmaps, query_extrinsic=None):
        '''
        :param xyz: [n_samples, 3]
        :param intrinsics: [n_views, 4, 4] intrinsic of source images
        :param extrinsics: [n_views, 4, 4] extrinsic of source images
        :param images: [images, 3, H, W] source images
        :param featmaps: [n_views, d, h, w] feature maps of source images
        :param query_extrinsic: [4,4] extrinsic of target camera
        :return: feat_sampled: [n_samples, n_views, n_feat],
                 mask: [n_samples, n_views, 1]
        '''
        # assert (train_imgs.shape[0] == 1) \
        #        and (train_cameras.shape[0] == 1), 'only support batch_size=1 for now'

        # train_cameras = train_cameras.squeeze(0)  # [n_views, 34]

        h, w = images.shape[-2:]

        # compute the projection of the query points to each reference image
        pixel_locations, depths, mask_in_front = self.compute_projections(xyz, intrinsics, extrinsics)
        normalized_pixel_locations = self.normalize(pixel_locations, h, w)   # [n_views, n_samples, 2]

        # rgb sampling
        rgbs_sampled = F.grid_sample(images, normalized_pixel_locations.unsqueeze(2), align_corners=True)
        rgbs_sampled = rgbs_sampled.squeeze(-1).permute(2,0,1)  # [n_samples, n_views, 3]

        # deep feature sampling
        feat_sampled = F.grid_sample(featmaps, normalized_pixel_locations.unsqueeze(2), align_corners=True) # [n_views, d, n_samples, 1]
        feat_sampled = feat_sampled.squeeze(-1).permute(2,0,1)  # [n_samples, n_views, d]

        # mask
        inbound = self.inbound(pixel_locations, h, w)
        mask = (inbound * mask_in_front).float().permute(1,0)[..., None]   # [n_samples, n_views, 1]
        if query_extrinsic is not None:
            ray_diff = self.compute_angle(xyz, query_extrinsic, extrinsics)
            ray_diff = ray_diff.permute(1,0,2) # [n_samples, n_views, 4]
            return rgbs_sampled, feat_sampled, ray_diff, mask
        return rgbs_sampled, feat_sampled, mask

class FeatureAggregator(nn.Module):
    def __init__(self, args, in_feat_ch, out_feat_ch, in_appearance_ch, hidden_dim=32, activation_func=nn.ELU(inplace=True)):
        super().__init__()
        self.anti_alias_pooling = args.anti_alias_pooling
        if self.anti_alias_pooling:
            self.s = nn.Parameter(torch.tensor(0.2), requires_grad=True)
        self.ray_dir_fc = nn.Sequential(nn.Linear(4, 16),
                                        activation_func,
                                        nn.Linear(16, in_feat_ch + 3),
                                        activation_func)

        self.base_fc = nn.Sequential(nn.Linear((in_feat_ch+3)*3, 64),
                                     activation_func,
                                     nn.Linear(64, 32),
                                     activation_func)

        self.vis_fc = nn.Sequential(nn.Linear(32, 32),
                                    activation_func,
                                    nn.Linear(32, 33),
                                    activation_func,
                                    )

        self.vis_fc2 = nn.Sequential(nn.Linear(32, 32),
                                     activation_func,
                                     nn.Linear(32, 1),
                                     nn.Sigmoid()
                                     )

        self.geometry_fc = nn.Sequential(nn.Linear(32*2+1, 64),
                                         activation_func,
                                         nn.Linear(64, 16),
                                         activation_func)

        self.ray_attention = MultiHeadAttention(4, 16, 4, 4)
        self.out_geometry_fc = nn.Sequential(nn.Linear(16, 16),
                                             activation_func,
                                             nn.Linear(16, 1),
                                             nn.ReLU())

        self.rgb_fc = nn.Sequential(nn.Linear(32+1+4, 16),
                                    activation_func,
                                    nn.Linear(16, 8),
                                    activation_func,
                                    nn.Linear(8, 1))

        self.agg_weights_fc = nn.Sequential(
            # nn.Linear(3*hidden_dim+1, hidden_dim),
            nn.Linear(3*32+1, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        self.desc_fc = nn.Sequential(
            nn.Linear(3+in_feat_ch+in_appearance_ch, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, hidden_dim),
            activation_func,
            nn.Linear(hidden_dim, out_feat_ch),
        )

        self.base_fc.apply(weights_init)
        self.vis_fc.apply(weights_init)
        self.vis_fc2.apply(weights_init)
        self.geometry_fc.apply(weights_init)
        self.rgb_fc.apply(weights_init)
        self.agg_weights_fc.apply(weights_init)
        self.desc_fc.apply(weights_init)

    def ray_pos_encoding(self, d_hid, n_samples, device):

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        # TODO:
        sinusoid_table = torch.from_numpy(sinusoid_table).to(device).float().unsqueeze(0)
        return sinusoid_table

    def forward(self, rgb_feat, mask, appearance_embed=None, ray_diff=None, n_rays=None, n_samples=None):
        """
        Args: 
            rgb_feat: image features [N,V,3+C]
            mask: mask for whether each projection is valid or not. [N,V,1]
            ray_diff: ray direction difference [N, V, 4], first 3 channels are directions,
        Returns: 
            desc_3d: N,C
        """        
        num_views = rgb_feat.shape[1]
        rgb_in = rgb_feat[...,:3]
        if ray_diff is not None:
            direction_feat = self.ray_dir_fc(ray_diff)
        else:
            direction_feat = 0
        if self.anti_alias_pooling:
            _, dot_prod = torch.split(ray_diff, [3, 1], dim=-1)
            exp_dot_prod = torch.exp(torch.abs(self.s) * (dot_prod - 1))
            weight = (exp_dot_prod - torch.min(exp_dot_prod, dim=1, keepdim=True)[0]) * mask
            weight = weight / (torch.sum(weight, dim=1, keepdim=True) + 1e-8)
        else:
            weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)
        # weight = mask / (torch.sum(mask, dim=1, keepdim=True) + 1e-8)

        rgb_feat = rgb_feat + direction_feat


        # compute mean and variance across different views for each point
        mean, var = fused_mean_variance(rgb_feat, weight)  # [N, 1, 3+C]
        globalfeat = torch.cat([mean, var], dim=-1)  # [N, 1, 2*(3+C)]

        x = torch.cat([globalfeat.expand(-1, num_views, -1), rgb_feat], dim=-1)  # [N, V, 3*(3+C)]
        x = self.base_fc(x)

        x_vis = self.vis_fc(x * weight)
        x_res, vis = torch.split(x_vis, [x_vis.shape[-1]-1, 1], dim=-1)
        vis = torch.sigmoid(vis) * mask
        x = x + x_res
        vis = self.vis_fc2(x * vis) * mask
        weight = vis / (torch.sum(vis, dim=1, keepdim=True) + 1e-8) # [N,V,1]

        if n_rays is not None and n_samples is not None:
            # density
            mean, var = fused_mean_variance(x, weight) # [N, 1, C]
            globalfeat = torch.cat([mean.squeeze(1), var.squeeze(1), weight.mean(dim=1)], dim=-1)  # [N, C*2+1]
            globalfeat = self.geometry_fc(globalfeat)  # [N, 16]
            num_valid_obs = torch.sum(mask, dim=1) # [N,1]
            pos_emb = self.ray_pos_encoding(globalfeat.shape[-1], n_samples, globalfeat.device) # [1,n_samples,C]
            globalfeat = (globalfeat.view(n_rays, n_samples, -1) + pos_emb).view(n_rays, n_samples, -1)
            num_valid_obs = num_valid_obs.view(n_rays, n_samples, -1)
            globalfeat, _ = self.ray_attention(globalfeat, globalfeat, globalfeat,
                                            mask=(num_valid_obs > 1).float())  # [n_rays, n_samples, 16]
            sigma = self.out_geometry_fc(globalfeat)  # [n_rays, n_samples, 1]
            sigma_out = sigma.masked_fill(num_valid_obs < 1, 0.)  # set the sigma of invalid point to zero

            # rgb computation
            x = torch.cat([x, vis, ray_diff], dim=-1)
            x = self.rgb_fc(x)
            x = x.masked_fill(mask == 0, -1e9)
            blending_weights_valid = F.softmax(x, dim=1)  # color blending

            rgb_out = torch.sum(rgb_in*blending_weights_valid, dim=1).view(n_rays, n_samples, 3)
            out = torch.cat([rgb_out, sigma_out], dim=-1)
            return out
        else:
            # desc
            mean, var = fused_mean_variance(x, weight) # [N, 1, C]
            x = torch.cat([x, mean.repeat(1,num_views,1), var.repeat(1,num_views,1), vis], dim=-1) # N,V,3*C+1
            x = self.agg_weights_fc(x)
            x = x.masked_fill(mask == 0, -1e9)
            agg_weights_valid = torch.softmax(x, dim=1)
            feature_agg = torch.sum(rgb_feat*agg_weights_valid, dim=1)

            # # simply average
            # feature_agg = torch.sum(rgb_feat*mask, dim=1)

            if appearance_embed is not None:
                desc_3d = self.desc_fc(torch.cat([
                    feature_agg,
                    appearance_embed], dim=-1))
            else:
                desc_3d = self.desc_fc(feature_agg)
            ################
            return desc_3d
