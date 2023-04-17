"""
Author: jenningsliu
Date: 2022-06-01 22:19:57
LastEditors: jenningsliu
LastEditTime: 2022-08-16 17:16:36
FilePath: /nerf-loc/pl/test.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import argparse
import os
from configs import get_cfg_defaults, override_cfg_with_args
import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import Model
from datasets import build_dataset

from models.nerf_pose_estimator import NerfPoseEstimator

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument("--num_nodes", type=int, default=1, help='number of nodes')
    parser.add_argument("--gpus", type=int, default=-1, help='number of gpus')
    parser.add_argument("--ckpt", type=str, default=None, help='whole model file')
    parser.add_argument("--test_render_interval", type=int, default=-1, help='interval of rendering test image')
    parser.add_argument('--vis_3d_box', action='store_true', help='save onepose box visualization')
    parser.add_argument('--vis_rendering', action='store_true', help='save rendered image for visualization')
    parser.add_argument('--vis_trajectory', action='store_true', help='save camera trajectory for visualization')
    args = parser.parse_args()

    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg = override_cfg_with_args(cfg, args)

    basedir = cfg.basedir
    expname = cfg.expname
    exp_dir = os.path.join(basedir, expname)
    os.makedirs(exp_dir, exist_ok=True)

    # logger = TensorBoardLogger(save_dir=basedir, name=expname, version=cfg.version)

    trainset = build_dataset(cfg, 'train')
    # trainset.set_mode('test')

    testset = build_dataset(cfg, 'test')
    testset.set_mode('test')
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=1, num_workers=8, shuffle=False, drop_last=False)

    pose_estimator = NerfPoseEstimator(cfg, trainset)
    model = Model(pose_estimator).eval()
    trainer = pl.Trainer(     
        # logger=logger,   
        accelerator='ddp',
        num_nodes=1,
        devices=1,
        gpus=-1,
        max_epochs=1000,
        num_sanity_val_steps=20,
        benchmark=True,
    )

    if args.ckpt is not None:
        model.load_ckpt(args.ckpt)
    trainer.test(model, dataloaders=test_dataloader)
