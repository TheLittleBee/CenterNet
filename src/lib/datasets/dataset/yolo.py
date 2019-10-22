from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import cv2
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math
import warnings

import torch.utils.data as data


class YOLO(data.Dataset):
    default_resolution = [416, 416]
    mean = np.array([0.40789654, 0.44719302, 0.47026115],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.28863828, 0.27408164, 0.27809835],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, data_dir, hflip, vflip, rotation, scale, shear, opt, split, gray=False):
        # 读文件得到路径和标注
        with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
            self.class_name = [l.rstrip() for l in f.readlines()]
        self.num_classes = len(self.class_name)
        if split == 'train':
            with open(os.path.join(data_dir, 'train.txt'), 'r') as f:
                self.images = [l.rstrip() for l in f.readlines()]
        elif split == 'val':
            with open(os.path.join(data_dir, 'valid.txt'), 'r') as f:
                self.images = [l.rstrip() for l in f.readlines()]
        else:
            with open(os.path.join(data_dir, 'test.txt'), 'r') as f:
                self.images = [l.rstrip() for l in f.readlines()]
        self.anno = [l.replace('images', 'labels').replace('.jpg', '.txt') for l in self.images]

        self.hflip = hflip
        self.vflip = vflip
        self.rotation = rotation
        self.scale = scale
        self.shear = shear
        self.gray = gray
        self.opt = opt
        self.split = split
        self.max_objs = 128

    def __getitem__(self, index):
        img_id = self.images[index]
        img = cv2.imread(img_id)
        height, width = img.shape[0], img.shape[1]
        # YOLO标注转换
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            anns = np.loadtxt(self.anno[index]).reshape(-1, 5)
        if anns.size:
            x1 = width * (anns[:, 1] - anns[:, 3] / 2)
            y1 = height * (anns[:, 2] - anns[:, 4] / 2)
            x2 = width * (anns[:, 1] + anns[:, 3] / 2)
            y2 = height * (anns[:, 2] + anns[:, 4] / 2)
            anns[:, 1] = x1
            anns[:, 2] = y1
            anns[:, 3] = x2
            anns[:, 4] = y2
        num_objs = min(len(anns), self.max_objs)

        # 数据变换
        c = np.array([width / 2., height / 2.], dtype=np.float32)
        s = max(height, width) * 1.0
        rotation = 0
        shear = 0
        input_h, input_w = self.opt.input_h, self.opt.input_w

        hflipped = False
        vflipped = False
        if self.split == 'train':
            if self.shear:
                shear = np.clip(np.random.randn() * self.shear, -self.shear, self.shear)
            if shear:
                if shear < 0:
                    img = img[:, ::-1, :]
                    anns[:, [1, 3]] = width - anns[:, [3, 1]] - 1

                M = np.array([[1, abs(shear), 0], [0, 1, 0]])

                nW = width + abs(shear * height)

                anns[:, [1, 3]] += ((anns[:, [2, 4]]) * abs(shear)).astype(int)

                img = cv2.warpAffine(img, M, (int(nW), height))

                if shear < 0:
                    img = img[:, ::-1, :]
                    anns[:, [1, 3]] = nW - anns[:, [3, 1]] - 1
                c[0] = nW / 2.
                s = max(nW, s)
                width = nW

            sf = self.scale
            s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if self.hflip and np.random.random() < self.hflip:
                hflipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1
            if self.vflip and np.random.random() < self.vflip:
                vflipped = True
                img = img[::-1, :, :]
                c[1] = height - c[1] - 1
            # 旋转参数设置
            if self.rotation:
                rotation = np.clip(np.random.randn() * self.rotation, -self.rotation, self.rotation)


        trans_input = get_affine_transform(
            c, s, rotation, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        inp = (inp - self.mean) / self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, rotation, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            bbox = anns[k, 1:]
            cls_id = int(anns[k, 0])
            if hflipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            if vflipped:
                bbox[[1, 3]] = height - bbox[[3, 1]] - 1
            lt = affine_transform(bbox[:2], trans_output)
            rb = affine_transform(bbox[2:], trans_output)
            rt = affine_transform(bbox[[2, 1]], trans_output)
            lb = affine_transform(bbox[[0, 3]], trans_output)
            bbox[:2] = np.min([lt, rb, rt, lb], axis=0)
            bbox[2:] = np.max([lt, rb, rt, lb], axis=0)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret

    def run_eval(self, results, save_dir):
        dir_path = os.path.join(save_dir, 'result')
        if not os.path.exists(dir_path):
            os.mkdir(dir_path)
        for k, v in results.items():
            path = os.path.join(dir_path, os.path.basename(k).replace('.jpg', '.txt'))
            dets = np.zeros((0, 6), dtype=np.float32)
            for cls, det in v.items():
                cls_id = np.ones((len(det), 1), dtype=np.float32) * (cls - 1)
                dets = np.append(dets, np.hstack((det, cls_id)), 0)
            np.savetxt(path, dets)

    def __len__(self):
        return len(self.images)
