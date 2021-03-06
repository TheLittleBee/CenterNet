﻿import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
import cv2
import numpy as np
from tqdm import tqdm
from PyQt5.QtCore import pyqtSignal, QObject
import time

from models.model import create_model, load_model, save_model
from datasets.dataset.yolo import YOLO
from trains.base_trainer import ModleWithLoss
from trains.ctdet import CtdetLoss
from utils.utils import AverageMeter
from utils.debugger import Debugger
from utils.logger import setup_logger
from models.decode import ctdet_decode

USE_TENSORBOARD = True
try:
    import tensorboardX

    print('Using tensorboardX')
except:
    USE_TENSORBOARD = False


class CenterNet(QObject):
    def __int__(self):
        print('Welcome to train CenterNet model')
        # self.lossSignal = pyqtSignal(float, int)

    def train(self, cfg):
        # 设置gpu环境,考虑单卡多卡情况
        gpus_str = ''
        if isinstance(cfg.gpus, (list, tuple)):
            cfg.gpus = [int(i) for i in cfg.gpus]
            for s in cfg.gpus: gpus_str += str(s)+','
            gpus_str = gpus_str[:-1]
        else:
            gpus_str = str(int(cfg.gpus))
            cfg.gpus = [int(cfg.gpus)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str
        cfg.gpus = [i for i in range(len(cfg.gpus))] if cfg.gpus[0] >= 0 else [-1]

        # 设置log
        model_dir = os.path.join(cfg.save_dir, cfg.id)
        debug_dir = os.path.join(model_dir, 'debug')
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        if not os.path.exists(debug_dir):
            os.makedirs(debug_dir)
        logger = setup_logger(cfg.id, os.path.join(model_dir, 'log'))
        if USE_TENSORBOARD:
            writer = tensorboardX.SummaryWriter(log_dir=os.path.join(model_dir, 'log'))
        logger.info(cfg)

        gpus = cfg.gpus
        device = torch.device('cpu' if gpus[0] < 0 else 'cuda')
        lr = cfg.lr
        lr_step = cfg.lr_step
        num_epochs = cfg.num_epochs
        val_step = cfg.val_step
        sample_size = cfg.sample_size

        # 设置数据集
        dataset = YOLO(cfg.data_dir, cfg.hflip, cfg.vflip, cfg.rotation, cfg.scale, cfg.shear, opt=cfg, split='train')
        names = dataset.class_name
        std = dataset.std
        mean = dataset.mean
        # 用数据集类别数设置预测网络
        cfg.setup_head(dataset)
        trainloader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                                 num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

        # val_dataset = YOLO(cfg.data_dir, cfg.hflip, cfg.vflip, cfg.rotation, cfg.scale, cfg.shear, opt=cfg, split='val')
        # valloader = DataLoader(val_dataset, batch_size=1, shuffle=True, num_workers=1, pin_memory=True)
        valid_file = cfg.val_dir if not cfg.val_dir == '' else os.path.join(cfg.data_dir, 'valid.txt')
        with open(valid_file, 'r') as f:
            val_list = [l.rstrip() for l in f.readlines()]

        net = create_model(cfg.arch, cfg.heads, cfg.head_conv, cfg.down_ratio, cfg.filters)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        start_epoch = 0

        if cfg.resume:
            pretrain = os.path.join(model_dir, 'model_last.pth')
            if os.path.exists(pretrain):
                print('resume model from %s' % pretrain)
                try:
                    net, optimizer, start_epoch = load_model(net, pretrain, optimizer, True, lr, lr_step)
                except:
                    print('\t... loading model error: ckpt may not compatible')
        model = ModleWithLoss(net, CtdetLoss(cfg))
        if len(gpus)>1:
            model = nn.DataParallel(model, device_ids=gpus).to(device)
        else:
            model = model.to(device)

        step = 0
        best = 1e10
        log_loss_stats = ['loss', 'hm_loss', 'wh_loss']
        if cfg.reg_offset:
            log_loss_stats += ['off_loss']
        if cfg.reg_obj:
            log_loss_stats += ['obj_loss']
        for epoch in range(start_epoch + 1, num_epochs + 1):
            avg_loss_stats = {l: AverageMeter() for l in log_loss_stats}
            model.train()
            with tqdm(trainloader) as loader:
                for _, batch in enumerate(loader):
                    for k in batch:
                        if k != 'meta':
                            batch[k] = batch[k].to(device=device, non_blocking=True)
                    output, loss, loss_stats = model(batch)
                    loss = loss.mean()
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    # 设置tqdm显示信息
                    lr = optimizer.param_groups[0]['lr']
                    poststr = ''
                    for l in avg_loss_stats:
                        avg_loss_stats[l].update(
                            loss_stats[l].mean().item(), batch['input'].size(0))
                        poststr += '{}: {:.4f}; '.format(l, avg_loss_stats[l].avg)
                    loader.set_description('Epoch %d' % (epoch))
                    poststr += 'lr: {:.4f}'.format(lr)
                    loader.set_postfix_str(poststr)

                    step += 1
                    # self.lossSignal.emit(loss.item(), step)
                    del output, loss, loss_stats

                    # valid
                    if step % val_step == 0:
                        if len(cfg.gpus) > 1:
                            val_model = model.module
                        else:
                            val_model = model
                        val_model.eval()
                        torch.cuda.empty_cache()

                        # 随机采样
                        idx = np.arange(len(val_list))
                        idx = np.random.permutation(idx)[:sample_size]

                        for j, id in enumerate(idx):
                            image = cv2.imread(val_list[id])
                            image = self.preprocess(image, cfg.input_h, cfg.input_w, mean, std)
                            image = image.to(device)

                            with torch.no_grad():
                                output = val_model.model(image)[-1]

                            # 画图并保存
                            debugger = Debugger(dataset=names, down_ratio=cfg.down_ratio)
                            reg = output['reg'] if cfg.reg_offset else None
                            obj = output['obj'] if cfg.reg_obj else None
                            dets = ctdet_decode(
                                output['hm'].sigmoid_(), output['wh'], reg=reg, obj=obj,
                                cat_spec_wh=cfg.cat_spec_wh, K=cfg.K)
                            dets = dets.detach().cpu().numpy().reshape(-1, dets.shape[2])
                            dets[:, :4] *= cfg.down_ratio
                            image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
                            image = np.clip(((image * std + mean) * 255.), 0, 255).astype(np.uint8)
                            pred = debugger.gen_colormap(output['hm'][0].detach().cpu().numpy())
                            debugger.add_blend_img(image, pred, 'pred_hm')
                            debugger.add_img(image, img_id='out_pred')
                            for k in range(len(dets)):
                                if dets[k, 4] > cfg.vis_thresh:
                                    debugger.add_coco_bbox(dets[k, :4], dets[k, -1],
                                                           dets[k, 4], img_id='out_pred')

                            debugger.save_all_imgs(debug_dir, prefix='{}.{}_'.format(step, j))
                            del output, image, dets
                        # 保存模型参数
                        save_model(os.path.join(model_dir, 'model_best.pth'), epoch, net)
                        model.train()

            logstr = 'epoch {}'.format(epoch)
            for k, v in avg_loss_stats.items():
                logstr += ' {}: {:.4f};'.format(k, v.avg)
                if USE_TENSORBOARD:
                    writer.add_scalar('train_{}'.format(k), v.avg, epoch)
            logger.info(logstr)

            # if epoch % val_step == 0:
            #     if len(cfg.gpus) > 1:
            #         val_model = model.module
            #     else:
            #         val_model = model
            #     val_model.eval()
            #     torch.cuda.empty_cache()
            #
            #     val_loss_stats = {l: AverageMeter() for l in log_loss_stats}
            #
            #     with tqdm(valloader) as loader:
            #         for j, batch in enumerate(loader):
            #             for k in batch:
            #                 if k != 'meta':
            #                     batch[k] = batch[k].to(device=device, non_blocking=True)
            #             with torch.no_grad():
            #                 output, loss, loss_stats = val_model(batch)
            #
            #             poststr = ''
            #             for l in val_loss_stats:
            #                 val_loss_stats[l].update(
            #                     loss_stats[l].mean().item(), batch['input'].size(0))
            #                 poststr += '{}: {:.4f}; '.format(l, val_loss_stats[l].avg)
            #             loader.set_description('Epoch %d valid' % (epoch))
            #             poststr += 'lr: {:.4f}'.format(lr)
            #             loader.set_postfix_str(poststr)
            #
            #             if j < sample_size:
            #                 # 将预测结果画出来保存成jpg图片
            #                 debugger = Debugger(dataset=names, down_ratio=cfg.down_ratio)
            #                 reg = output['reg'] if cfg.reg_offset else None
            #                 obj = output['obj'] if cfg.reg_obj else None
            #                 dets = ctdet_decode(
            #                     output['hm'], output['wh'], reg=reg, obj=obj,
            #                     cat_spec_wh=cfg.cat_spec_wh, K=cfg.K)
            #                 dets = dets.detach().cpu().numpy().reshape(1, -1, dets.shape[2])
            #                 dets[:, :, :4] *= cfg.down_ratio
            #                 dets_gt = batch['meta']['gt_det'].numpy().reshape(1, -1, dets.shape[2])
            #                 dets_gt[:, :, :4] *= cfg.down_ratio
            #                 for i in range(1):
            #                     img = batch['input'][i].detach().cpu().numpy().transpose(1, 2, 0)
            #                     img = np.clip(((img * std + mean) * 255.), 0, 255).astype(np.uint8)
            #                     pred = debugger.gen_colormap(output['hm'][i].detach().cpu().numpy())
            #                     gt = debugger.gen_colormap(batch['hm'][i].detach().cpu().numpy())
            #                     debugger.add_blend_img(img, pred, 'pred_hm')
            #                     debugger.add_blend_img(img, gt, 'gt_hm')
            #                     debugger.add_img(img, img_id='out_pred')
            #                     for k in range(len(dets[i])):
            #                         if dets[i, k, 4] > cfg.vis_thresh:
            #                             debugger.add_coco_bbox(dets[i, k, :4], dets[i, k, -1],
            #                                                    dets[i, k, 4], img_id='out_pred')
            #
            #                     debugger.add_img(img, img_id='out_gt')
            #                     for k in range(len(dets_gt[i])):
            #                         if dets_gt[i, k, 4] > cfg.vis_thresh:
            #                             debugger.add_coco_bbox(dets_gt[i, k, :4], dets_gt[i, k, -1],
            #                                                    dets_gt[i, k, 4], img_id='out_gt')
            #
            #                     debugger.save_all_imgs(debug_dir, prefix='{}.{}_'.format(epoch, j))
            #             del output, loss, loss_stats
            #     model.train()
            #     logstr = 'epoch {} valid'.format(epoch)
            #     for k, v in val_loss_stats.items():
            #         logstr += ' {}: {:.4f};'.format(k, v.avg)
            #         if USE_TENSORBOARD:
            #             writer.add_scalar('val_{}'.format(k), v.avg, epoch)
            #     logger.info(logstr)
            #     if val_loss_stats['loss'].avg < best:
            #         best = val_loss_stats['loss'].avg
            #         save_model(os.path.join(model_dir, 'model_best.pth'), epoch, net)
            save_model(os.path.join(model_dir, 'model_last.pth'), epoch, net, optimizer)
            if epoch in cfg.lr_step:
                save_model(os.path.join(model_dir, 'model_{}.pth'.format(epoch)),
                           epoch, net, optimizer)
                lr = cfg.lr * (0.1 ** (cfg.lr_step.index(epoch) + 1))
                logger.info('Drop LR to {}'.format(lr))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

    @staticmethod
    def preprocess(image, input_h, input_w, mean, std):
        """预处理模块,resize至模型输入大小,并归一化"""
        from utils.image import get_affine_transform

        height, width = image.shape[0:2]
        inp_height, inp_width = input_h, input_w
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image = cv2.warpAffine(
            image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - mean) / std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        return images
