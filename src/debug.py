from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

import numpy as np

from opts import opts
from datasets.dataset.yolo import YOLO
from utils.debugger import Debugger

if __name__ == '__main__':
    opt = opts().parse()
    dataset = YOLO(opt.data_dir, opt.flip, opt.vflip, opt.rotate, opt.scale, opt, 'train')
    opt = opts().update_dataset_info_and_set_heads(opt, dataset)
    for i in range(len(dataset)):
        debugger = Debugger(dataset=dataset)
        data = dataset[i]
        img = data['input'].transpose(1, 2, 0)
        hm = data['hm']
        dets_gt = data['meta']['gt_det']
        dets_gt[:, :4] *= opt.down_ratio
        img = np.clip(((img * dataset.std + dataset.mean) * 255.), 0, 255).astype(np.uint8)
        pred = debugger.gen_colormap(hm)
        debugger.add_blend_img(img, pred, 'pred_hm')
        debugger.add_img(img, img_id='out_pred')
        for k in range(len(dets_gt)):
            debugger.add_coco_bbox(dets_gt[k, :4], dets_gt[k, -1],dets_gt[k, 4], img_id='out_pred')
        debugger.show_all_imgs(pause=True)
