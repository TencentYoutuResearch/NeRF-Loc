import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from kornia.geometry.subpix import dsnt
from kornia.utils.grid import create_meshgrid
from einops.einops import rearrange

class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        in_channels_coarse = config['in_channels_coarse']
        in_channels_fine = config['in_channels_fine']
        out_channels = config['out_channels']
        self.out_channels = out_channels
        if self.cat_c_feat:
            self.down_proj = nn.Linear(in_channels_coarse, in_channels_fine, bias=True)
            self.merge_feat = nn.Linear(2*in_channels_fine, out_channels, bias=True)
        else:
            self.proj = nn.Linear(in_channels_fine, out_channels, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f1, feat_c1, data):
        """
        Args: 
            feat_f1: (B,C,Hf,Wf)
            feat_c1: coarse feature map (B,C,Hc,Wc)
        Returns: 
        """        
        W = self.W
        stride = data['stride_coarse'] // data['stride_fine']

        # data.update({'W': W})
        if len(data['j_ids']) == 0:
            feat1 = torch.empty(0, self.W**2, self.out_channels, device=feat_f1.device)
            return feat1

        # 1. unfold(crop) all local windows
        # feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        # feat_f0_unfold = rearrange(feat_f0_unfold, 'n (c ww) l -> n l ww c', ww=W**2)
        feat_f1_unfold = F.unfold(feat_f1, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f1_unfold = rearrange(feat_f1_unfold, 'n (c ww) l -> n l ww c', ww=W**2)

        # 2. select only the predicted matches
        feat_f1_unfold = feat_f1_unfold[data['b_ids'], data['j_ids']] # k,ww,c
        # # option: use coarse-level loftr feature as context: concat and linear
        if self.cat_c_feat:
            feat_c1_pick = feat_c1.view(feat_c1.shape[0], feat_c1.shape[1], -1)[data['b_ids'], :, data['j_ids']] # k,c'
            feat_c1_down = self.down_proj(feat_c1_pick)
            feat_f1_unfold = self.merge_feat(torch.cat([
                feat_c1_down.unsqueeze(1).repeat(1,W**2,1),
                feat_f1_unfold
            ], dim=2))
            # feat_c_win = self.down_proj(torch.cat([feat_c0[data['b_ids'], data['i_ids']],
            #                                        feat_c1[data['b_ids'], data['j_ids']]], 0))  # [2n, c]
            # feat_cf_win = self.merge_feat(torch.cat([
            #     torch.cat([feat_f0_unfold, feat_f1_unfold], 0),  # [2n, ww, cf]
            #     repeat(feat_c_win, 'n c -> n ww c', ww=W**2),  # [2n, ww, cf]
            # ], -1))
            # feat_f0_unfold, feat_f1_unfold = torch.chunk(feat_cf_win, 2, dim=0)
        else:
            feat_f1_unfold = self.proj(feat_f1_unfold)

        return feat_f1_unfold


class FineMatching(nn.Module):
    """FineMatching with s2d paradigm"""

    def __init__(self, config):
        super().__init__()
        self.correct_thr = config['correct_thr']
        self.loss_type = config['loss_type']
        self.mlps = nn.Sequential(
            nn.Linear(config['feat_dim'], 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, feat_f0, feat_f1, data):
        """
        Args:
            feat0 (torch.Tensor): [M, C]
            feat1 (torch.Tensor): [M, WW, C]
            data (dict)
        Update:
            data (dict):{
                'expec_f' (torch.Tensor): [M, 3],
                'mkps2d_f' (torch.Tensor): [M, 2]}
        """
        M, WW, C = feat_f1.shape
        W = int(math.sqrt(WW)) # window size
        # scale = data['hw0_i'][0] / data['hw0_f'][0]
        scale = data['stride_fine'] # to input scale
        self.M, self.W, self.WW, self.C, self.scale = M, W, WW, C, scale

        # corner case: if no coarse matches found
        if M == 0:
            assert self.training == False, "M is always >0, when training, see coarse_matching.py"
            # logger.warning('No matches found in coarse-level.')
            data.update({
                'expec_f': torch.empty(0, 3, device=feat_f0.device),
                'mkps2d_f': data['mkps2d_c'],
            })
            return

        # sim_matrix = torch.einsum('mc,mrc->mr', feat_f0, feat_f1)
        sim_matrix = torch.einsum('mc,mrc->mrc', feat_f0, feat_f1)
        sim_matrix = self.mlps(sim_matrix).squeeze(-1) # mr
        softmax_temp = 1. / C**.5
        heatmap = torch.softmax(softmax_temp * sim_matrix, dim=1).view(-1, W, W)

        # compute coordinates from heatmap
        coords_normalized = dsnt.spatial_expectation2d(heatmap[None], True)[0]  # [M, 2]
        grid_normalized = create_meshgrid(W, W, True, heatmap.device).reshape(1, -1, 2)  # [1, WW, 2]

        # compute std over <x, y>
        var = torch.sum(grid_normalized**2 * heatmap.view(-1, WW, 1), dim=1) - coords_normalized**2  # [M, 2]
        std = torch.sum(torch.sqrt(torch.clamp(var, min=1e-10)), -1)  # [M]  clamp needed for numerical stability
        
        # for fine-level supervision
        data.update({'expec_f': torch.cat([coords_normalized, std.unsqueeze(1)], -1)})

        if self.training:
            data['fine_loss'] = self.get_loss(data['expec_f'], data['expec_f_gt']) # expec_f_gt is normalized

        # compute absolute kpt coords
        self.get_fine_match(coords_normalized, data)
        return data

    @torch.no_grad()
    def get_fine_match(self, coords_normed, data):
        W, WW, C, scale = self.W, self.WW, self.C, self.scale

        mkps2d_f = data['mkps2d_c'] + coords_normed * (W // 2)

        data.update({
            "mkps2d_f": mkps2d_f
        })

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss

    def get_loss(self, expec_f, expec_f_gt):
        if self.loss_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.loss_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()
