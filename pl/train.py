"""
Author: jenningsliu
Date: 2022-06-01 22:19:57
LastEditors: jenningsliu
LastEditTime: 2022-08-19 19:00:51
FilePath: /nerf-loc/pl/train.py
Description: 
Copyright (c) 2022 by Tencent, All Rights Reserved. 
"""
import os
import glob
import argparse

import torch
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from model import Model
from datasets import build_dataset

from models.nerf_pose_estimator import NerfPoseEstimator
from configs import get_cfg_defaults, override_cfg_with_args

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help='config file path')
    parser.add_argument("--num_nodes", type=int, default=1, help='number of nodes')
    parser.add_argument("--gpus", type=int, default=-1, help='number of gpus')
    args = parser.parse_args()
    
    cfg = get_cfg_defaults()
    cfg.merge_from_file(args.config)
    cfg = override_cfg_with_args(cfg, args)

    basedir = cfg.basedir
    expname = cfg.expname
    exp_dir = os.path.join(basedir, expname)
    os.makedirs(exp_dir, exist_ok=True)
    # logging.basicConfig(level=logging.INFO, filename=os.path.join(exp_dir, 'train_pose.log'))
    # logging.basicConfig(level=logging.INFO, stream=sys.stdout)
    # logger = logging.getLogger(__name__)
    # logger.info(str(cfg))
    logger = TensorBoardLogger(save_dir=basedir, name=expname, version=cfg.version)

    trainset = build_dataset(cfg, 'train')
    trainset.set_mode('train')
    train_size = len(trainset)
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=1, num_workers=10, shuffle=True, drop_last=False, pin_memory=True)

    testset = build_dataset(cfg, 'test')
    testset.set_mode('test')
    test_dataloader = torch.utils.data.DataLoader(
        testset, batch_size=1, num_workers=10, shuffle=False, drop_last=False, pin_memory=True)

    if cfg.dataset_type == 'nerf_pretrain' or not cfg.train_pose:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(exp_dir, cfg.version, 'checkpoints'),
            filename='epoch{epoch:02d}-psnr{psnr_test:.4f}',
            monitor='psnr_test',
            mode='max',
            save_top_k=1,
            verbose=True,
            auto_insert_metric_name=False
        )
    elif cfg.dataset_type.startswith('video_') and testset.datasets[0].cfg.type == 'cambridge':
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(exp_dir, cfg.version, 'checkpoints'),
            filename='epoch{epoch:02d}-median_trans_err{median_trans_err/avg:.4f}',
            monitor='median_trans_err/avg',
            mode='min',
            save_top_k=5,
            verbose=True,
            auto_insert_metric_name=False
        )
    else:
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(exp_dir, cfg.version, 'checkpoints'),
            filename='epoch{epoch:02d}-acc{pose_acc/avg:.4f}',
            monitor='pose_acc/avg',
            mode='max',
            save_last=True,
            save_top_k=5,
            verbose=True,
            auto_insert_metric_name=False
        )

    pose_estimator = NerfPoseEstimator(cfg, trainset)
    print(pose_estimator)
    model = Model(pose_estimator)

    # auto resume from the last checkpoint
    ckpts = glob.glob(os.path.join(exp_dir, cfg.version, 'checkpoints')+'/*.ckpt')
    ckpts = sorted(ckpts)
    if len(ckpts) > 0:
        resume_from_checkpoint = ckpts[-1]
        print('resume from ', resume_from_checkpoint)
    else:
        resume_from_checkpoint = None

    trainer = pl.Trainer(     
        logger=logger,   
        callbacks=[checkpoint_callback],
        accelerator='ddp',
        num_nodes=args.num_nodes,
        gpus=args.gpus,
        max_epochs=cfg.max_epochs,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        benchmark=True,
        gradient_clip_val=1.0,
        resume_from_checkpoint=resume_from_checkpoint
    )
    if resume_from_checkpoint is None and cfg.ckpt is not None:
        print('load pretrain weights: ', cfg.ckpt)
        model.load_ckpt(cfg.ckpt)
    
    trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_dataloader)

    # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_dataloader, ckpt_path=cfg.ckpt)

    # state_dict = {k.replace('pose_estimator.', ''):v for k,v in torch.load(cfg.ckpt)['state_dict'].items()}
    # model.pose_estimator.load_state_dict(state_dict)
    # # trainer.fit(model=model, train_dataloaders=train_loader, val_dataloaders=test_dataloader)
    # trainer.test(model, dataloaders=test_dataloader)
