"""
Author: jenningsliu
Date: 2022-06-01 19:22:02
LastEditors: jenningsliu
LastEditTime: 2022-08-19 19:28:05
FilePath: /nerf-loc/pl/model.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import os
from collections import defaultdict
import numpy as np
import torch
import pytorch_lightning as pl
from torchmetrics import MeanMetric
import cv2
import pickle
import itertools

from nerf_loc.utils.metrics import compute_pose_error, compute_matching_iou
from nerf_loc.utils.common import colorize
from nerf_loc.utils.visualization import draw_onepose_3d_box
from nerf_loc.models.nerf import img2mse, mse2psnr

# define the LightningModule
class Model(pl.LightningModule):
    def __init__(self, pose_estimator):
        super().__init__()
        self.pose_estimator = pose_estimator
        # self.mean_rot_error = MeanMetric()
        # self.mean_trans_error = MeanMetric()

    def load_ckpt(self, ckpt):
        print(f'Resume from {ckpt}')
        state_dict = {k.replace('pose_estimator.', ''):v for k,v in torch.load(ckpt)['state_dict'].items()}
        current_params_shapes = {k:v.shape for k,v in self.pose_estimator.named_parameters()}
        matched_state_dict = {}
        for k,v in state_dict.items():
            if k in current_params_shapes and current_params_shapes[k] == v.shape:
                matched_state_dict[k] = v
        self.pose_estimator.load_state_dict(matched_state_dict, strict=False)

    def training_step(self, batch, batch_idx):
        # img = batch['image']
        pose = batch['pose']
        # topk_poses = batch['topk_poses']
        # topk_idxs = batch['topk_idxs']
        # K = batch['K']
        # depth = batch['depth']
        # points3d = batch['points3d']

        ret = self.pose_estimator(batch)
        loss = ret['loss']
        if self.pose_estimator.args.train_pose:
            coarse_loss = ret['coarse_match_loss']
            fine_loss = ret['fine_match_loss']
            fine_error = ret['fine_err']
            pos_num = len(ret['pairs_gt'][0])
            m_iou = compute_matching_iou(ret['pairs'], ret['pairs_gt'])
            matched_num = len(ret['pairs'][0])
            # rot_err, trans_err = compute_pose_error(ret['T'], pose[0].cpu().numpy())
            # trans_err /= batch['scale_factor'][0] # back to original scale
            # print(m_iou, rot_err, trans_err)
            # Logging to TensorBoard by default
            # self.log("train_loss", loss)
            self.log_dict({
                'loss': loss,
                # 'desc_3d_fine_loss': ret['desc_3d_fine_loss'],
                'coarse_match_loss': ret['coarse_match_loss'],
                'fine_match_loss': ret['fine_match_loss'],
                'fine_err': ret['fine_err'],
                'ref_depth_loss': ret.get('ref_depth_loss', -1),
                'match_iou': m_iou,
                'pos_num': pos_num,
                'matched_num': matched_num,
                # 'inter_match_loss': ret['inter_match_loss'],
            }, on_step=True, on_epoch=False)
            if self.pose_estimator.args.optimize_pose:
                self.log_dict({
                    'pose_refine_loss': ret['pose_refine_loss'],
                    'trans_err_refine': ret['trans_err_refine'],
                    'rot_err_refine': ret['rot_err_refine']
                }, on_step=True, on_epoch=False)
        if self.pose_estimator.args.train_nerf:
            self.log_dict({
                'render_loss': ret['render_loss'],
                'psnr_train': ret['psnr']
            }, on_step=True, on_epoch=False)

        return loss

    def test_step(self, batch, batch_idx):
        # img = batch['image']
        pose = batch['pose']
        # topk_poses = batch['topk_poses']
        # topk_idxs = batch['topk_idxs']
        # K = batch['K']
        # depth = batch['depth']
        # points3d = batch['points3d']
        args = self.pose_estimator.args
        if args.train_nerf and args.test_render_interval > 0 and batch_idx % args.test_render_interval == 0:
            batch['render_image'] = True

        outputs = {}
        ret = self.pose_estimator(batch)
        if args.train_pose:
            m_iou = compute_matching_iou(ret['pairs'], ret['pairs_gt'])
            mkps2d = ret['mkps2d']
            matched_num = len(mkps2d)
            pos_num = len(ret['pairs_gt'][0])
            rot_err, trans_err = compute_pose_error(ret['T'], pose[0].cpu().numpy())
            trans_err /= batch['scale_factor'][0].item() # back to original scale
            outputs = {
                'scene': batch['scene'][0],
                'match_iou': m_iou,
                'rot_err': rot_err,
                'trans_err': trans_err,
                'T': ret['T'],
                'T_gt': pose[0].cpu().numpy(),
                'filename': batch['filename'][0]
            }

        if 'rendered_image' in ret:
            max_depth = batch['far'][0].detach().cpu().numpy()
            rendered_img = (ret['rendered_image']*255).cpu().numpy().astype(np.uint8)
            rendered_depth = (colorize(
                ret['rendered_depth'].cpu().squeeze(-1), range=[0, max_depth]
            )*255).cpu().numpy().astype(np.uint8)
            # rendered_depth = ((ret['rendered_depth'].cpu().squeeze(-1))*255).cpu().numpy().astype(np.uint8)
            src_images = (batch['topk_images'][0]*255).permute(0,2,3,1).cpu().numpy().astype(np.uint8)
            gt_img = (batch['image'][0]*255).permute(1,2,0).cpu().numpy().astype(np.uint8)
            gt_depth = (colorize(
                batch['depth'].squeeze().cpu(), range=[0, max_depth]
            )*255).cpu().numpy().astype(np.uint8)

            self.logger.experiment.add_image('render_image', rendered_img, batch_idx, dataformats='HWC')
            self.logger.experiment.add_image('gt_image', gt_img, batch_idx, dataformats='HWC')
            self.logger.experiment.add_image('render_depth', rendered_depth, batch_idx, dataformats='HWC')
            self.logger.experiment.add_image('gt_depth', gt_depth, batch_idx, dataformats='HWC')
            if 'rendered_depth_coarse' in ret:
                rendered_depth_coarse = (colorize(
                    ret['rendered_depth_coarse'].cpu().squeeze(-1), range=[0, max_depth]
                )*255).cpu().numpy().astype(np.uint8)
                self.logger.experiment.add_image(
                    'rendered_depth_coarse', rendered_depth_coarse, batch_idx, dataformats='HWC')
            self.logger.experiment.flush()
            psnr = mse2psnr(img2mse(
                ret['rendered_image'], batch['image'][0].permute(1,2,0), 
                mask=batch['target_mask'][0] if 'target_mask' in batch else None))
            # self.log_dict({
            #     'psnr_test': psnr
            # }, on_step=True, on_epoch=False)
            outputs.update({'psnr': psnr})
            if args.vis_rendering:
                vis_rendering_dir = os.path.join(args.basedir, args.expname, args.version, 'vis_rendering')
                os.makedirs(os.path.join(vis_rendering_dir, 'pred'), exist_ok=True)
                os.makedirs(os.path.join(vis_rendering_dir, 'gt'), exist_ok=True)
                out_filename = '_'.join(batch['filename'][0].split('/')[-2:])
                cv2.imwrite(
                    os.path.join(vis_rendering_dir, 'pred', out_filename), 
                    cv2.cvtColor(rendered_img, cv2.COLOR_RGB2BGR))
                cv2.imwrite(
                    os.path.join(vis_rendering_dir, 'gt', out_filename), 
                    cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))

        if args.vis_3d_box:
            log_dir = os.path.join(args.basedir, args.expname, args.version)
            image = (batch['image'][0]*255).permute(1,2,0).contiguous().cpu().numpy().astype(np.uint8)
            w2c = np.linalg.inv(ret['T'])
            w2c_gt = np.linalg.inv(pose[0].cpu().numpy())
            K = batch['K'][0].cpu().numpy()
            box_corners = batch['bbox3d_corners'][0].cpu().numpy()
            image = draw_onepose_3d_box(box_corners, image, w2c_gt, K, color=(0,255,0)) # ground truth
            image = draw_onepose_3d_box(box_corners, image, w2c, K, color=(0,0,255)) # prediction
            out_filename = os.path.join(log_dir, 'vis_3d_box', batch['filename'][0])
            os.makedirs(os.path.dirname(out_filename), exist_ok=True)
            cv2.imwrite(out_filename, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

        return outputs

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def predict_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    # def training_epoch_end(self, outs):
    #     sch = self.lr_schedulers()
    #     sch.step()
    #     self.log('lr', sch.get_last_lr()[0])

    def test_epoch_end(self, outs):
        # gather outputs from all process
        torch.distributed.barrier()
        gather = [None] * torch.distributed.get_world_size()
        torch.distributed.all_gather_object(gather, outs)
        outs = list(itertools.chain(*gather)) # flatten

        print(f'Total length of predictions: {len(outs)}')
        args = self.pose_estimator.args
        if args.train_nerf:
            test_psnr = [out['psnr'].item() for out in outs if 'psnr' in out]
            if len(test_psnr) > 0:
                self.log_dict({'psnr_test': np.array(test_psnr).mean()}, on_step=False, on_epoch=True)
        if not args.train_pose:
            return
        rot_errs = defaultdict(list)
        trans_errs = defaultdict(list)
        matching_iou = defaultdict(list)
        if args.vis_trajectory:
            trajectories = defaultdict(list)
        # if self.pose_estimator.cascade_matching:
        #     coarse_rot_errs = defaultdict(list)
        #     coarse_trans_errs = defaultdict(list)
        for out in outs:
            scene = out['scene']
            rot_errs[scene].append(out['rot_err'])
            trans_errs[scene].append(out['trans_err'])
            matching_iou[scene].append(out['match_iou'])
            if args.vis_trajectory:
                trajectories[scene].append({
                    'filename': out['filename'],
                    'T': out['T'],
                    'T_gt': out['T_gt']
                })
            # if self.pose_estimator.cascade_matching:
            #     coarse_rot_errs[scene].append(out['coarse_rot_err'])
            #     coarse_trans_errs[scene].append(out['coarse_trans_err'])
        
        CAMBRIDGE_THRESH = {
            'StMarysChurch': 0.35,
            'GreatCourt': 0.45,
            'OldHospital': 0.22,
            'KingsCollege': 0.38,
            'ShopFacade': 0.15
        }
        average_median_trans_err = 0
        average_pose_acc = 0
        for scene in rot_errs.keys():
            rot_correct = np.array(rot_errs[scene])<self.pose_estimator.args.rotation_eval_thresh
            translation_eval_thresh = CAMBRIDGE_THRESH.get(scene, self.pose_estimator.args.translation_eval_thresh)
            trans_correct = np.array(trans_errs[scene])<translation_eval_thresh
            rot_acc = rot_correct.sum() / len(rot_errs[scene])
            trans_acc = trans_correct.sum() / len(trans_errs[scene])
            pose_acc = (rot_correct & trans_correct).sum() / len(trans_errs[scene])
            # self.log(f'rot_acc/{scene}', rot_acc, on_step=False, on_epoch=True)
            # self.log(f'trans_acc/{scene}', trans_acc, on_step=False, on_epoch=True)
            self.log(f'pose_acc/{scene}', pose_acc, on_step=False, on_epoch=True)

            rot_errs[scene] = np.median(rot_errs[scene])
            trans_errs[scene] = np.median(trans_errs[scene])
            matching_iou[scene] = np.array(matching_iou[scene]).mean()
            self.log(f'median_rot_err/{scene}', rot_errs[scene], on_step=False, on_epoch=True)
            self.log(f'median_trans_err/{scene}', trans_errs[scene], on_step=False, on_epoch=True)
            self.log(f'mean_matching_iou/{scene}', matching_iou[scene], on_step=False, on_epoch=True)
            average_median_trans_err += trans_errs[scene]
            average_pose_acc += pose_acc

        n_scenes = len(rot_errs)
        self.log('median_trans_err/avg', average_median_trans_err/n_scenes)
        self.log('pose_acc/avg', average_pose_acc/n_scenes)

        if args.vis_trajectory and torch.distributed.get_rank() == 0:
            # save trajectories for visualization
            log_dir = os.path.join(args.basedir, args.expname, args.version)
            os.makedirs(log_dir, exist_ok=True)
            print('Saved trajectories to '+ os.path.join(log_dir, 'trajectories.pkl'))
            with open(os.path.join(log_dir, 'trajectories.pkl'), 'wb') as fout:
                pickle.dump(trajectories, fout)
            if args.dataset_type.startswith('video_'):
                scene_points = {}
                for ds in self.pose_estimator.dataset.datasets:
                    scene_points[ds.scene] = np.array(ds.pc.vertices)
                print('Saved scene_points to '+ os.path.join(log_dir, 'scene_points.pkl'))
                with open(os.path.join(log_dir, 'scene_points.pkl'), 'wb') as fout:
                    pickle.dump(scene_points, fout)

    def validation_epoch_end(self, outs):
        self.test_epoch_end(outs)

    def on_after_backward(self):
        valid_gradients = True
        for name, param in self.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    # print(name)
                    break

        if not valid_gradients:
            # print(f'Detected inf or nan values in gradients, not updating model parameters')
            self.zero_grad()

    def configure_optimizers(self):
        params = [p for p in self.pose_estimator.parameters() if p.requires_grad]
        params_name = [n for n,p in self.pose_estimator.named_parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params=params, lr=self.pose_estimator.args.lrate, betas=(0.9, 0.999))
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
            step_size=self.pose_estimator.args.lrate_decay_steps, gamma=self.pose_estimator.args.lrate_decay_factor)
        return [optimizer], [scheduler]
