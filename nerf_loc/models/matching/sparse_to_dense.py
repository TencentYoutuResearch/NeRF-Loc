"""
Author: jenningsliu
Date: 2022-03-08 15:47:46
LastEditors: jenningsliu
LastEditTime: 2022-05-14 15:19:08
FilePath: /nerf-loc/models/matching/sparse_to_dense.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class SigmoidFocalClassificationLoss(nn.Module):
    """
    Sigmoid focal cross entropy loss.
    """

    def __init__(self, gamma: float = 2.0, alpha: float = 0.25):
        """
        Args:
            gamma: Weighting parameter to balance loss for hard and easy examples.
            alpha: Weighting parameter to balance loss for positive and negative examples.
        """
        super(SigmoidFocalClassificationLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def sigmoid_cross_entropy_with_logits(input: torch.Tensor, target: torch.Tensor):
        """ PyTorch Implementation for tf.nn.sigmoid_cross_entropy_with_logits:
            max(x, 0) - x * z + log(1 + exp(-abs(x))) in
            https://www.tensorflow.org/api_docs/python/tf/nn/sigmoid_cross_entropy_with_logits

        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets

        Returns:
            loss: (B, #anchors, #classes) float tensor.
                Sigmoid cross entropy loss without reduction
        """
        loss = torch.clamp(input, min=0) - input * target + \
               torch.log1p(torch.exp(-torch.abs(input)))
        return loss

    def forward(self, input: torch.Tensor, target: torch.Tensor, weights: torch.Tensor):
        """
        Args:
            input: (B, #anchors, #classes) float tensor.
                Predicted logits for each class
            target: (B, #anchors, #classes) float tensor.
                One-hot encoded classification targets
            weights: (B, #anchors) float tensor.
                Anchor-wise weights.

        Returns:
            weighted_loss: (B, #anchors, #classes) float tensor after weighting.
        """
        pred_sigmoid = torch.sigmoid(input)
        alpha_weight = target * self.alpha + (1 - target) * (1 - self.alpha)
        pt = target * (1.0 - pred_sigmoid) + (1.0 - target) * pred_sigmoid
        focal_weight = alpha_weight * torch.pow(pt, self.gamma)

        bce_loss = self.sigmoid_cross_entropy_with_logits(input, target)
        # bce_loss = F.binary_cross_entropy_with_logits(input, target, reduction='none')

        loss = focal_weight * bce_loss

        if weights.shape.__len__() == 2 or \
                (weights.shape.__len__() == 1 and target.shape.__len__() == 2):
            weights = weights.unsqueeze(-1)

        assert weights.shape.__len__() == loss.shape.__len__()

        return loss * weights

class S2DMatching(nn.Module):
    def __init__(self, feat_dim, thr=0.1):
        super().__init__()
        self.mlps = nn.Sequential(
            nn.Linear(feat_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )
        self.thr = thr
        self.focal_loss = SigmoidFocalClassificationLoss()

    def get_loss(self, conf, conf_gt):
        """
        Args: 
            conf: N,M
            conf_gt: N,M
        Returns: 
        """        
        # CE
        # loss = F.binary_cross_entropy_with_logits(conf, conf_gt, reduction='mean')

        # # focal loss
        # pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        # conf = torch.clamp(conf, 1e-6, 1-1e-6)
        # alpha = 0.25
        # gamma = 2.0
        # loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
        # loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
        # loss = loss_pos.mean() + loss_neg.mean()

        loss = self.focal_loss(conf.unsqueeze(2), conf_gt.unsqueeze(2), torch.ones_like(conf_gt.unsqueeze(2))).mean()

        return loss

    def forward(self, desc0, desc1, data):
        """
        Args: 
            desc0: (N,C) sparse set of descriptors
            desc1: (M,C) dense set of descriptors, where M >> N
        Returns: 
        """        
        assert (desc0.shape[0] > 0) and (desc1.shape[0] > 0)
        # x = torch.einsum('NC,MC->NMC', desc0, desc1)
        x = torch.einsum('nc,mc->nmc', desc0, desc1)
        conf_matrix = self.mlps(x).squeeze(-1) # N,M
        score_matrix = torch.sigmoid(conf_matrix)

        # i_ids = torch.arange(len(desc0), device=desc0.device)
        # max_v, j_ids = score_matrix.max(dim=1)
        # valid_mask = max_v > self.thr
        # i_ids = i_ids[valid_mask]
        # j_ids = j_ids[valid_mask]

        # mutual nearest
        mask = score_matrix > self.thr
        mask = mask \
            * (score_matrix == score_matrix.max(dim=1, keepdim=True)[0]) \
            * (score_matrix == score_matrix.max(dim=0, keepdim=True)[0])
        mask_v, all_j_ids = mask.max(dim=1) # N
        i_ids = torch.where(mask_v)[0]
        j_ids = all_j_ids[i_ids] # N'
        data.update({
            'i_ids': i_ids,
            'j_ids': j_ids,
            'score_matrix': score_matrix
        })
        if self.training:
            data['coarse_loss'] = self.get_loss(conf_matrix, data['conf_matrix_gt'].float())

        return data
