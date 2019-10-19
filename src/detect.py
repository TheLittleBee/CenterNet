from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import os
import cv2
import numpy as np
import time
from progress.bar import Bar
import torch

from opts import opts
from logger import Logger
from utils.utils import AverageMeter
from detectors.detector_factory import detector_factory
from datasets.dataset.yolo import YOLO
from utils.debugger import Debugger


def detect(opt):
    os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str

    split = 'val' if not opt.trainval else 'test'
    dataset = YOLO(opt.data_dir, opt.flip, opt.vflip, opt.rotate, opt.scale, opt.shear, opt, split)

    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    print(opt)
    # log = Logger(opt)
    Detector = detector_factory[opt.task]
    detector = Detector(opt)
    debugger = Debugger(dataset=opt.names)

    dir_path = os.path.join(opt.save_dir, 'detect')
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)

    images = []
    if os.path.isfile(opt.image):
        if os.path.splitext(opt.image)[1] == '.txt':
            name = os.path.splitext(os.path.basename(opt.image))[0]
            dir_path = os.path.join(dir_path, name)
            if not os.path.exists(dir_path):
                os.mkdir(dir_path)
            with open(opt.image, 'r') as f:
                images.extend([l.rstrip() for l in f.readlines()])
        elif os.path.splitext(opt.image)[1] in ['.jpg', '.png', '.bmp']:
            images.append(opt.image)
        else:
            raise Exception('NOT SUPPORT FILE TYPE!!!')
    else:
        for file in os.listdir(opt.image):
            if os.path.splitext(file)[1] in ['.jpg', '.png', '.bmp']:
                images.append(os.path.join(opt.image, file))
    num_iters = len(images)
    bar = Bar('{}'.format(opt.exp_id), max=num_iters)
    time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']
    avg_time_stats = {t: AverageMeter() for t in time_stats}
    for ind in range(num_iters):
        img_id = images[ind]
        ret = detector.run(img_id)

        Bar.suffix = '[{0}/{1}]|Tot: {total:} |ETA: {eta:} '.format(
            ind, num_iters, total=bar.elapsed_td, eta=bar.eta_td)

        for t in avg_time_stats:
            avg_time_stats[t].update(ret[t])
            Bar.suffix = Bar.suffix + '|{} {:.3f} '.format(t, avg_time_stats[t].avg)
        bar.next()

        img_name = os.path.splitext(os.path.basename(img_id))[0]
        debugger.add_img(cv2.imread(img_id), img_id=img_name)
        path = os.path.join(dir_path, os.path.basename(img_id).replace('.jpg', '.txt'))
        dets = np.zeros((0, 6), dtype=np.float32)
        for cls, det in ret['results'].items():
            cls_id = np.ones((len(det), 1), dtype=np.float32) * (cls - 1)
            dets = np.append(dets, np.hstack((det, cls_id)), 0)
            for d in det:
                debugger.add_coco_bbox(d[:4], cls-1, d[-1], img_id=img_name)
        np.savetxt(path, dets)
    bar.finish()
    debugger.save_all_imgs(path=dir_path)
    # log.close()

if __name__ == '__main__':
    opt = opts().parse()
    detect(opt)
