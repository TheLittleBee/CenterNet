from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import torch

try:
    from external.nms import soft_nms
except:
    raise Exception('nms needed!!! Please do \n cd $CenterNet_ROOT/src/lib/external \n make')
from models.decode import fcos_decode

from .base_detector import BaseDetector


class FCOSDetector(BaseDetector):
    def __init__(self, opt):
        super(FCOSDetector, self).__init__(opt)

    def process(self, images, return_time=False):
        with torch.no_grad():
            output = self.model(images)[-1]
            cls, reg, centerness = output
            cls = [c.sigmoid_() for c in cls]
            if len(centerness): centerness = [c.sigmoid_() for c in centerness]
            torch.cuda.synchronize()
            forward_time = time.time()
            strides = [2, 4, 8, 16, 32]
            if len(centerness):
                dets = [fcos_decode(c, r, s, n, K=100) for c, r, s, n in zip(cls, reg, strides, centerness)]
            else:
                dets = [fcos_decode(c, r, s, K=100) for c, r, s in zip(cls, reg, strides)]
            dets = torch.cat(dets, dim=1)

        if return_time:
            return output, dets, forward_time
        else:
            return output, dets

    def post_process(self, dets, meta, scale=1):
        ret = {}
        for j in range(1, self.num_classes + 1):
            ret[j] = dets[dets[..., -1] == j - 1][..., :-1].cpu().numpy().reshape(-1, 5)    # pytorch version incompatible
            soft_nms(ret[j], Nt=0.5, method=2)
            ret[j][:, :4] /= scale
        return ret

    def merge_outputs(self, detections):
        results = {}
        for j in range(1, self.num_classes + 1):
            results[j] = np.concatenate(
                [detection[j] for detection in detections], axis=0).astype(np.float32)
            if len(self.scales) > 1 or self.opt.nms:
                soft_nms(results[j], Nt=0.5, method=2)
        scores = np.hstack(
            [results[j][:, 4] for j in range(1, self.num_classes + 1)])
        if len(scores) > self.max_per_image:
            kth = len(scores) - self.max_per_image
            thresh = np.partition(scores, kth)[kth]
            for j in range(1, self.num_classes + 1):
                keep_inds = (results[j][:, 4] >= thresh)
                results[j] = results[j][keep_inds]
        return results

    def debug(self, debugger, images, dets, output, scale=1):
        detection = dets.detach().cpu().numpy().copy()
        for i in range(1):
            img = images[i].detach().cpu().numpy().transpose(1, 2, 0)
            img = ((img * self.std + self.mean) * 255).astype(np.uint8)
            debugger.add_img(img, img_id='out_pred_{:.1f}'.format(scale))
            for k in range(len(dets[i])):
                if detection[i, k, 4] > self.opt.center_thresh:
                    debugger.add_coco_bbox(detection[i, k, :4], detection[i, k, -1],
                                           detection[i, k, 4],
                                           img_id='out_pred_{:.1f}'.format(scale))

    def show_results(self, debugger, image, results):
        debugger.add_img(image, img_id='ctdet')
        for j in range(1, self.num_classes + 1):
            for bbox in results[j]:
                if bbox[4] > self.opt.vis_thresh:
                    debugger.add_coco_bbox(bbox[:4], j - 1, bbox[4], img_id='ctdet')
        debugger.show_all_imgs(pause=self.pause)
