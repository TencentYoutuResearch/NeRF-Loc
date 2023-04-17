import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from inplace_abn import ABN

from .neuray_ops import ResEncoder, interpolate_feats, masked_mean_var

def coords2rays(coords, poses, Ks):
    """
    :param coords:   [rfn,rn,2]
    :param poses:    [rfn,3,4]
    :param Ks:       [rfn,3,3]
    :return:
        ref_rays:
            centers:    [rfn,rn,3]
            directions: [rfn,rn,3]
    """
    rot = poses[:, :, :3].unsqueeze(1).permute(0, 1, 3, 2)  # rfn,1,3,3
    trans = -rot @ poses[:, :, 3:].unsqueeze(1)  # rfn,1,3,1

    rfn, rn, _ = coords.shape
    centers = trans.repeat(1, rn, 1, 1).squeeze(-1)  # rfn,rn,3
    coords = torch.cat([coords, torch.ones([rfn, rn, 1], dtype=torch.float32, device=coords.device)], 2)  # rfn,rn,3
    Ks_inv = torch.inverse(Ks).unsqueeze(1)
    cam_xyz = Ks_inv @ coords.unsqueeze(3)
    cam_xyz = rot @ cam_xyz + trans
    directions = cam_xyz.squeeze(3) - centers
    # directions = directions / torch.clamp(torch.norm(directions, dim=2, keepdim=True), min=1e-4)
    return centers, directions

def depth2points(que_imgs_info, que_depth):
    """
    :param que_imgs_info:
    :param que_depth:       qn,rn,dn
    :return:
    """
    cneters, directions = \
        coords2rays(que_imgs_info['coords'],que_imgs_info['poses'],que_imgs_info['Ks']) # centers, directions qn,rn,3
    qn, rn, _ = cneters.shape
    que_pts = cneters.unsqueeze(2) + directions.unsqueeze(2) * que_depth.unsqueeze(3) # qn,rn,dn,3
    qn, rn, dn, _ = que_pts.shape
    que_dir = -directions / torch.norm(directions, dim=2, keepdim=True)  # qn,rn,3
    que_dir = que_dir.unsqueeze(2).repeat(1, 1, dn, 1)
    return que_pts, que_dir # qn,rn,dn,3

def depth2dists(depth):
    device = depth.device
    dists = depth[...,1:]-depth[...,:-1]
    return torch.cat([dists, torch.full([*depth.shape[:-1], 1], 1e6, dtype=torch.float32, device=device)], -1)

def depth2inv_dists(depth, depth_range):
    near, far = -1 / depth_range[:, 0], -1 / depth_range[:, 1]
    near, far = near[:, None, None], far[:, None, None]
    depth_inv = -1 / depth  # qn,rn,dn
    depth_inv = (depth_inv - near) / (far - near)
    dists = depth2dists(depth_inv)  # qn,rn,dn
    return dists

def interpolate_feature_map(ray_feats, coords, mask, h, w, border_type='border'):
    """
    :param ray_feats:       rfn,f,h,w
    :param coords:          rfn,pn,2
    :param mask:            rfn,pn
    :param h:
    :param w:
    :param border_type:
    :return:
    """
    fh, fw = ray_feats.shape[-2:]
    if fh == h and fw == w:
        cur_ray_feats = interpolate_feats(ray_feats, coords, h, w, border_type, True)  # rfn,pn,f
    else:
        cur_ray_feats = interpolate_feats(ray_feats, coords, h, w, border_type, False)  # rfn,pn,f
    cur_ray_feats = cur_ray_feats * mask.float().unsqueeze(-1) # rfn,pn,f
    return cur_ray_feats

def project_points_coords(pts, Rt, K):
    """
    :param pts:  [pn,3]
    :param Rt:   [rfn,3,4]
    :param K:    [rfn,3,3]
    :return:
        coords:         [rfn,pn,2]
        invalid_mask:   [rfn,pn]
    """
    pn = pts.shape[0]
    hpts = torch.cat([pts,torch.ones([pn,1],device=pts.device,dtype=torch.float32)],1)
    srn = Rt.shape[0]
    KRt = K @ Rt # rfn,3,4
    last_row = torch.zeros([srn,1,4],device=pts.device,dtype=torch.float32)
    last_row[:,:,3] = 1.0
    H = torch.cat([KRt,last_row],1) # rfn,4,4
    pts_cam = H[:,None,:,:] @ hpts[None,:,:,None]
    pts_cam = pts_cam[:,:,:3,0]
    depth = pts_cam[:,:,2:]
    invalid_mask = torch.abs(depth)<1e-4
    depth[invalid_mask] = 1e-3
    pts_2d = pts_cam[:,:,:2]/depth
    return pts_2d, ~(invalid_mask[...,0]), depth

def project_points_directions(poses,points):
    """
    :param poses:       rfn,3,4
    :param points:      pn,3
    :return: rfn,pn,3
    """
    cam_pts = -poses[:, :, :3].permute(0, 2, 1) @ poses[:, :, 3:]  # rfn,3,1
    dir = points.unsqueeze(0) - cam_pts.permute(0, 2, 1)  # [1,pn,3] - [rfn,1,3] -> rfn,pn,3
    dir = -dir / torch.clamp_min(torch.norm(dir, dim=2, keepdim=True), min=1e-5)  # rfn,pn,3
    return dir

def project_points_ref_views(ref_imgs_info, que_points):
    """
    :param ref_imgs_info:
    :param que_points:      pn,3
    :return:
    """
    prj_pts, prj_valid_mask, prj_depth = project_points_coords(
        que_points, ref_imgs_info['poses'], ref_imgs_info['Ks']) # rfn,pn,2
    h,w=ref_imgs_info['imgs'].shape[-2:]
    prj_img_invalid_mask = (prj_pts[..., 0] < -0.5) | (prj_pts[..., 0] >= w - 0.5) | \
                           (prj_pts[..., 1] < -0.5) | (prj_pts[..., 1] >= h - 0.5)
    valid_mask = prj_valid_mask & (~prj_img_invalid_mask)
    prj_dir = project_points_directions(ref_imgs_info['poses'], que_points) # rfn,pn,3
    return prj_dir, prj_pts, prj_depth, valid_mask

def project_points_dict(ref_imgs_info, que_pts):
    # que_pts: N,3
    # project all points
    # qn, rn, dn, _ = que_pts.shape
    n_pts = que_pts.shape[0]
    prj_dir, prj_pts, prj_depth, prj_mask = project_points_ref_views(ref_imgs_info, que_pts)
    rfn, _, h, w = ref_imgs_info['imgs'].shape
    prj_ray_feats = interpolate_feature_map(ref_imgs_info['ray_feats'], prj_pts, prj_mask, h, w)
    prj_rgb = interpolate_feature_map(ref_imgs_info['imgs'], prj_pts, prj_mask, h, w)
    prj_dict = \
        {
            'dir':prj_dir, 'pts':prj_pts, 'depth':prj_depth, 
            'mask': prj_mask.float(), 'ray_feats':prj_ray_feats, 'rgb':prj_rgb
        }

    # post process
    for k, v in prj_dict.items():
        # prj_dict[k]=v.reshape(rfn,qn,rn,dn,-1)
        prj_dict[k]=v.reshape(rfn,n_pts,-1)
    return prj_dict


def depth2pts3d(depth, ref_Ks, ref_poses):
    rfn, dn, h, w = depth.shape
    coords = torch.stack(torch.meshgrid(torch.arange(h), torch.arange(w)), -1).float().to(depth.device)
    coords = coords[:, :, (1, 0)]
    coords = coords.unsqueeze(0)  # 1,h,w,2
    coords = torch.cat([
        coords, torch.ones([1, h, w, 1], dtype=torch.float32, device=depth.device)
    ], -1).unsqueeze(-2)  # 1,h,w,1,3
    # rfn,h,w,dn,1 1,h,w,1,3
    pts3d = depth.permute(0, 2, 3, 1).unsqueeze(-1) * coords  # rfn,h,w,dn,3
    pts3d = pts3d.reshape(rfn, h * w * dn, 3).permute(0, 2, 1)  # rfn,3,h*w*dn
    pts3d = torch.inverse(ref_Ks) @ pts3d  # rfn
    R = ref_poses[:, :3, :3].permute(0, 2, 1)  # rfn,3,3
    t = -R @ ref_poses[:, :3, 3:]  # rfn,3,1
    pts3d = R @ pts3d + t  # rfn,3,h*w*dn
    return pts3d.permute(0, 2, 1)  # rfn,h*w*dn,3

def get_diff_feats(ref_imgs_info, depth_in):
    """
    Reproject all reference depths to each other, compute depth and color differences
    Args: 
        ref_imgs_info: dict
        depth_in: [rfn,1,h,w]
    Returns: 
    """    
    imgs = ref_imgs_info['imgs']  # rfn,3,h,w
    depth_range = ref_imgs_info['depth_range']
    near = depth_range[:, 0][:, None, None]  # rfn,1,1
    far = depth_range[:, 1][:, None, None]  # rfn,1,1
    near_inv, far_inv = -1 / near[..., None], -1 / far[..., None]
    depth_in = depth_in * (far_inv - near_inv) + near_inv
    depth = -1 / depth_in
    rfn, _, h, w = imgs.shape

    pts3d = depth2pts3d(depth, ref_imgs_info['Ks'], ref_imgs_info['poses'])
    _, pts2d, pts_dpt_prj, valid_mask = \
        project_points_ref_views(ref_imgs_info, pts3d.reshape(-1, 3)) # [rfn,rfn*h*w,2] [rfn,rfn*h*w] [rfn,rfn*h*w,1]
    pts_dpt_int = interpolate_feats(depth, pts2d, padding_mode='border', align_corners=True) # rfn,rfn*h*w,1
    pts_rgb_int = interpolate_feats(imgs, pts2d, padding_mode='border', align_corners=True) # rfn,rfn*h*w,3

    rgb_diff = torch.abs(pts_rgb_int - imgs.permute(0, 2, 3, 1).reshape(1, rfn * h * w, 3)) # rfn,rfn*h*w,3

    pts_dpt_int = torch.clamp(pts_dpt_int, min=1e-5)
    pts_dpt_prj = torch.clamp(pts_dpt_prj, min=1e-5)
    dpt_diff = torch.abs(-1 / pts_dpt_int + 1 / pts_dpt_prj)  # rfn,rfn*h*w,1
    near_inv, far_inv = -1 / near, -1 / far
    dpt_diff = dpt_diff / (far_inv - near_inv)
    dpt_diff = torch.clamp(dpt_diff, max=1.5)

    valid_mask = valid_mask.float().unsqueeze(-1)
    dpt_mean, dpt_var = masked_mean_var(dpt_diff, valid_mask, 0)  # 1,rfn,h,w,1
    rgb_mean, rgb_var = masked_mean_var(rgb_diff, valid_mask, 0)  # 1,rfn*h*w,3
    dpt_mean = dpt_mean.reshape(rfn, h, w, 1).permute(0, 3, 1, 2)  # rfn,1,h,w
    dpt_var = dpt_var.reshape(rfn, h, w, 1).permute(0, 3, 1, 2)  # rfn,1,h,w
    rgb_mean = rgb_mean.reshape(rfn, h, w, 3).permute(0, 3, 1, 2)  # rfn,3,h,w
    rgb_var = rgb_var.reshape(rfn, h, w, 3).permute(0, 3, 1, 2)  # rfn,3,h,w

    return torch.cat([rgb_mean, rgb_var, dpt_mean, dpt_var], 1)

def cross_frames_depth_validation(ref_imgs_info, depth):
    imgs = ref_imgs_info['imgs']  # rfn,3,h,w
    rfn, _, h, w = imgs.shape
    pts3d = depth2pts3d(depth, ref_imgs_info['Ks'], ref_imgs_info['poses'])
    _, pts2d, pts_dpt_prj, valid_mask = \
        project_points_ref_views(ref_imgs_info, pts3d.reshape(-1, 3)) # [rfn,rfn*h*w,2] [rfn,rfn*h*w] [rfn,rfn*h*w,1]
    pts_dpt_int = interpolate_feats(depth, pts2d, padding_mode='border', align_corners=True) # rfn,rfn*h*w,1
    pts_dpt_int = torch.clamp(pts_dpt_int, min=1e-5)
    pts_dpt_prj = torch.clamp(pts_dpt_prj, min=1e-5)
    dpt_diff = torch.abs(pts_dpt_prj - pts_dpt_int)  # rfn,rfn*h*w,1
    return dpt_diff.view(rfn, rfn, h, w)

def extract_depth_for_init_impl(depth_range,depth):
    rfn, _, h, w = depth.shape

    near = depth_range[:, 0][:, None, None, None]  # rfn,1,1,1
    far = depth_range[:, 1][:, None, None, None]  # rfn,1,1,1
    near_inv = -1 / near
    far_inv = -1 / far
    depth = torch.clamp(depth, min=1e-5)
    depth = -1 / depth
    depth = (depth - near_inv) / (far_inv - near_inv)
    depth = torch.clamp(depth, min=0, max=1.0)
    return depth

def extract_depth_for_init(ref_imgs_info):
    depth_range = ref_imgs_info['depth_range']  # rfn,2
    depth = ref_imgs_info['depth']  # rfn,1,h,w
    return extract_depth_for_init_impl(depth_range, depth)

class DepthFusionNet(nn.Module):
    default_cfg={}
    def __init__(self, cfg={}, in_channels=None):
        super().__init__()
        self.cfg = {**self.default_cfg,**cfg}
        self.fuse_net = ResEncoder()
        # self.fuse_net = nn.Sequential(
        #     nn.Conv2d(12, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(64, 32, kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.ReLU(inplace=True)
        # )
        self.depth_skip = nn.Sequential(
            nn.Conv2d(1, 8, 2, 2),
            nn.ReLU(True),
            nn.Conv2d(8, 16, 2, 2)
        )
        self.conv_out=nn.Conv2d(16+32,32,1,1)
        self.out_channels = 32

    def forward(self, imgs, feats, depths, Ks, poses, depth_range):
        """
        Args: 
        Returns: 
            [V,C,H/4,W/4]
        """        
        ref_imgs_info = {
            'depth': depths.unsqueeze(1),
            'imgs': imgs,
            'poses': poses.inverse()[:,:3], # w2c [V,3,4]
            'Ks': Ks,
            'depth_range': depth_range.view(1,2).repeat(imgs.shape[0], 1).float()
        }
        depth = extract_depth_for_init(ref_imgs_info)
        imgs = ref_imgs_info['imgs']
        diff_feats = get_diff_feats(ref_imgs_info,depth)
        # imgs [b,3,h,w] depth [b,1,h,w] diff_feats [b,8,h,w]
        feats = self.fuse_net(torch.cat([imgs, depth, diff_feats], 1))
        depth_feats = self.depth_skip(depth)
        return self.conv_out(torch.cat([depth_feats, feats],1))
