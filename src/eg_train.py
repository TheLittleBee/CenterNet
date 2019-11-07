from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from config import get_cfg, parse_args
from engine import CenterNet

if __name__ == '__main__':
    cfg = get_cfg()
    # args = parse_args()
    cfg.model_cfg = 'centernet_RD_34_fpn_obj_s2_2x.yaml'
    cfg.merge_from_file()
    # cfg.update_dict(args)
    # 设置cfg参数
    """
    for example
        cfg.down_ratio = 2
    """
    print(cfg)

    centernet = CenterNet()
    centernet.train(cfg)
