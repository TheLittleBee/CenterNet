from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths

from config import get_cfg, parse_args
from engine import CenterNet

if __name__ == '__main__':
    cfg = get_cfg()
    args = parse_args()
    cfg.merge_from_file(args.model_cfg)
    cfg.update_dict(args)
    print(cfg)

    centernet = CenterNet()
    centernet.train(cfg)
