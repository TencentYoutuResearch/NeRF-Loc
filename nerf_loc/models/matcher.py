import torch
import torch.nn as nn
import torch.nn.functional as F
from .COTR.transformer import SelfCrossTransformer
from .COTR.position_encoding import PositionEmbeddingSine
from .matching.sparse_to_dense import S2DMatching
from .matching.coarse_matching import CoarseMatching
from .matching.fine_matching import FinePreprocess, FineMatching

class Matcher(nn.Module):
    def __init__(self, args, hidden_dim, in_channels_coarse, in_channels_fine, fine_matching=True):
        super().__init__()
        self.coarse_transformer = SelfCrossTransformer(
            d_model=hidden_dim,
            dropout=0.1,
            nhead=8,
            dim_feedforward=512,
            num_encoder_layers=6,
            num_decoder_layers=6,
            return_intermediate_dec=False,
        )
        self.coarse_matcher = S2DMatching(hidden_dim, thr=0.2)
        # self.coarse_matcher = CoarseMatching({
        #     'thr': 0.2,
        #     # 'match_type': 'sinkhorn',
        #     'match_type': 'dual_softmax',
        #     'dsmax_temperature': 0.1,
        #     'skh_init_bin_score': 1.0,
        #     'skh_iters': 3,
        #     'skh_prefilter': False,
        #     # 'sparse_spvs': True
        #     'sparse_spvs': False
        # })

        self.fine_matching = fine_matching
        if fine_matching:
            self.pos_emd_2d_fn = PositionEmbeddingSine(hidden_dim // 2, normalize=True, sine_type='lin_sine')

            self.fine_window_size = 7
            self.fine_preprocess = FinePreprocess({
                'fine_concat_coarse_feat': False,
                'fine_window_size': self.fine_window_size,
                'in_channels_coarse': in_channels_coarse,
                'in_channels_fine': in_channels_fine,
                'out_channels': hidden_dim
            })
            self.fine_transformer = SelfCrossTransformer(
                d_model=hidden_dim,
                dropout=0.1,
                nhead=8,
                dim_feedforward=128,
                num_encoder_layers=6,
                num_decoder_layers=6,
                return_intermediate_dec=False,
            )
            self.fine_matcher = FineMatching({
                'feat_dim': hidden_dim,
                'correct_thr': 1.0,
                # 'loss_type' :'l2_with_std'
                'loss_type' : args.fine_matching_loss_type
            })

    def forward(self, data):
        # kps_3d, desc_3d, kps_2d, desc_2d, conf_matrix_gt
        # data = {'conf_matrix_gt': conf_matrix_gt}
        # coarse matching
        # desc_3d, desc_2d = self.coarse_transformer(
        desc_3d_trans_c, desc_2d_trans_c = self.coarse_transformer(
            data['desc_3d'][None,...], data['pos_emd_3d'][None,...], 
            data['desc_2d_coarse'][None,...], data['pos_emd_2d'][None,...])
        # data['desc_3d'], data['desc_2d_coarse'] = desc_3d[0], desc_2d[0]
        data = self.coarse_matcher(desc_3d_trans_c[0], desc_2d_trans_c[0], data)
        i_ids = data['i_ids']
        j_ids = data['j_ids'] # in coarse level
        data['b_ids'] = torch.zeros_like(i_ids)
        # mkps3d = data['kps3d'][i_ids]
        # mkps2d_c = data['kps2d'][j_ids]
        data.update({
            'mkps3d': data['kps3d'][i_ids],
            'mkps2d_c': data['kps2d'][j_ids],
            'pairs': [i_ids, j_ids]
        })
        if not self.fine_matching:
            return data

        # fine matching
        if self.training:
            # use GT coarse matching for training fine matching
            i_ids = data['pairs_gt'][0]
            j_ids = data['pairs_gt'][1]
            data.update({
                'b_ids': torch.zeros_like(i_ids),
                'i_ids': i_ids,
                'j_ids': j_ids,
                'mkps2d_c': data['kps2d'][j_ids],
                'mkps3d': data['kps3d'][i_ids],
            })
            # compute normalized offset
            data['expec_f_gt'] = (data['kps3d_proj_gt'][i_ids] - data['mkps2d_c']) / (self.fine_window_size // 2)

        M = len(i_ids)
        if M == 0: # no coarse matching at all
            assert self.training == False, "M is always >0, when training"
            data.update({
                'expec_f': torch.empty(0, 3, device=data['mkps2d_c'].device),
                'mkps2d_f': data['mkps2d_c'],
            })
            return data

        feat_fine = data['feat_fine'].permute(0,3,1,2) # B,C,H,W
        feat_coarse = data['feat_coarse'].permute(0,3,1,2) # B,C,H,W
        
        # in older version, we use desc_3d_trans_c here
        # matched_desc_3d = data['desc_3d'][i_ids][:,None,:] # M,C -> M,1,C
        matched_desc_3d = data['desc_3d_fine'][i_ids][:,None,:] # M,C -> M,1,C
        matched_pos_emd_3d = data['pos_emd_3d'][i_ids][:,None,:] # M,C -> M,1,C

        matched_desc_2d_fine = self.fine_preprocess(feat_fine, feat_coarse, data) # M,WW,C
        matched_pos_emd_2d_fine = self.pos_emd_2d_fn(matched_desc_2d_fine[...,0].view(M, self.fine_window_size, self.fine_window_size)).view(M, self.fine_window_size*self.fine_window_size, -1) # M,WW,C
        matched_desc_3d, matched_desc_2d_fine = self.fine_transformer(
            matched_desc_3d, matched_pos_emd_3d, # M,K,C
            matched_desc_2d_fine, matched_pos_emd_2d_fine) # M,WW,C

        data = self.fine_matcher(matched_desc_3d[:,0,:], matched_desc_2d_fine, data)

        if self.training:
            data['fine_err'] = torch.norm(data['expec_f_gt']-data['expec_f'][:,:2], dim=-1).mean() * (self.fine_window_size // 2) * data['stride_fine']

        # # DEBUG
        # data['mkps2d_f'] =  data['mkps2d_c'] + data['expec_f_gt'] * (self.fine_window_size // 2)
        return data
