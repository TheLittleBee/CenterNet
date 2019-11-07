import torch
import numpy as np
import cv2
import os

import src._init_paths
from src.lib.models.model import create_model, load_model
from src.lib.datasets.dataset.yolo import YOLO
from src.lib.utils.image import get_affine_transform, transform_preds
from src.lib.models.decode import ctdet_decode


class Demo:
    def __init__(self, cfg, model_path=None):
        # 设置gpu环境,考虑单卡多卡情况
        gpus_str = ''
        if isinstance(cfg.gpus, (list, tuple)):
            cfg.gpus = [int(i) for i in cfg.gpus]
            for s in cfg.gpus: gpus_str += str(s) + ','
            gpus_str = gpus_str[:-1]
        else:
            gpus_str = str(int(cfg.gpus))
            cfg.gpus = [int(cfg.gpus)]
        os.environ['CUDA_VISIBLE_DEVICES'] = gpus_str
        cfg.gpus = [i for i in range(len(cfg.gpus))] if cfg.gpus[0] >= 0 else [-1]
        self.cfg = cfg
        if cfg.gpus[0] >= 0:
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        # 通过数据集类别数设置预测网络,也可直接提供类别数
        dataset = YOLO(cfg.data_dir, cfg.hflip, cfg.vflip, cfg.rotation, cfg.scale, cfg.shear, opt=cfg, split='val')
        names = dataset.class_name
        std = dataset.std
        mean = dataset.mean
        cfg.setup_head(dataset)
        # cfg.setup_head(num_classes)

        self.model = create_model(cfg.arch, cfg.heads, cfg.head_conv, cfg.down_ratio)
        if model_path is None:
            model_path = os.path.join(cfg.save_dir, cfg.id, 'model_best.pth')
        self.model = load_model(self.model, model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        self.mean = np.array(mean, dtype=np.float32).reshape(1, 1, 3)
        self.std = np.array(std, dtype=np.float32).reshape(1, 1, 3)

    def run(self, image_or_path_or_tensor):
        """

        Args:
            image_or_path_or_tensor:
        Returns:
            dets: (x1,y1,x2,y2,conf,c)
        """
        pre_processed = False
        if isinstance(image_or_path_or_tensor, np.ndarray):
            image = image_or_path_or_tensor
        elif type(image_or_path_or_tensor) == type(''):
            image = cv2.imread(image_or_path_or_tensor)
        else:
            assert len(image_or_path_or_tensor.size()) == 4
            image = image_or_path_or_tensor
            meta = {'c': (image.size(3) / 2, image.size(2) / 2),
                    's': max(image.shape[-2:]),
                    'out_height': image.size(2) // self.cfg.down_ratio,
                    'out_width': image.size(3) // self.cfg.down_ratio}
            pre_processed = True

        if not pre_processed:
            image, meta = self.preprocess(image)
        image = image.to(self.device)

        with torch.no_grad():
            output = self.model(image)[-1]

        reg = output['reg'] if self.cfg.reg_offset else None
        obj = output['obj'] if self.cfg.reg_obj else None
        dets = ctdet_decode(
            output['hm'].sigmoid_(), output['wh'], reg=reg, obj=obj,
            cat_spec_wh=self.cfg.cat_spec_wh, K=self.cfg.K)
        dets = self.postprocess(dets, meta)

        return dets

    def preprocess(self, image):
        height, width = image.shape[0:2]
        # 使用模型输入分辨率还是保持原图分辨率
        if not self.cfg.keep_res:
            inp_height, inp_width = self.cfg.input_h, self.cfg.input_w
            c = np.array([width / 2., height / 2.], dtype=np.float32)
            s = max(height, width) * 1.0
        else:
            # 保证分辨率能被pad+1整除
            inp_height = (height | self.cfg.pad) + 1
            inp_width = (width | self.cfg.pad) + 1
            c = np.array([width // 2, height // 2], dtype=np.float32)
            s = np.array([inp_width, inp_height], dtype=np.float32)

        trans_input = get_affine_transform(c, s, 0, [inp_width, inp_height])
        inp_image = cv2.warpAffine(
            image, trans_input, (inp_width, inp_height),
            flags=cv2.INTER_LINEAR)
        inp_image = ((inp_image / 255. - self.mean) / self.std).astype(np.float32)

        images = inp_image.transpose(2, 0, 1).reshape(1, 3, inp_height, inp_width)
        images = torch.from_numpy(images)
        # 记录变换参数,用于检测结果反变换
        meta = {'c': c, 's': s,
                'out_height': inp_height // self.cfg.down_ratio,
                'out_width': inp_width // self.cfg.down_ratio}
        return images, meta

    def postprocess(self, dets, meta):
        dets = dets.detach().cpu().numpy()
        dets = dets.reshape(-1, dets.shape[2])
        w, h = meta['out_width'], meta['out_height']
        dets[:, :2] = transform_preds(
            dets[:, 0:2], meta['c'], meta['s'], (w, h))
        dets[:, 2:4] = transform_preds(
            dets[:, 2:4], meta['c'], meta['s'], (w, h))
        return dets


if __name__ == '__main__':
    from src.lib.config.config import get_cfg

    cfg = get_cfg()
    cfg.merge_from_file()
    print(cfg)
    image_path = ''
    result = Demo(cfg).run(image_path)
