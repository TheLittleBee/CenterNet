import yaml
import argparse
import os

__all__ = ['get_cfg', 'parse_args']


class Cfg():
    def __init__(self):
        self.id = ''
        self.save_dir = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'exp', 'ctdet')
        self.model = dict()
        self.datasets = dict()
        self.trainer = dict()
        self.tester = dict()

    def convert(self):
        self.__dict__.update(self.model)
        self.__dict__.update(self.datasets)
        self.__dict__.update(self.trainer)
        self.__dict__.update(self.tester)

    def merge_from_file(self, file):
        with open(file, 'r') as fp:
            loaded_cfg = yaml.load(fp)
        self.id = loaded_cfg['id']
        if 'model' in loaded_cfg.keys():
            self.model.update(loaded_cfg['model'])
        if 'datasets' in loaded_cfg.keys():
            self.datasets.update(loaded_cfg['datasets'])
        if 'trainer' in loaded_cfg.keys():
            self.trainer.update(loaded_cfg['trainer'])
        if 'tester' in loaded_cfg.keys():
            self.tester.update(loaded_cfg['tester'])
        self.convert()
        delattr(self, 'model')
        delattr(self, 'datasets')
        delattr(self, 'trainer')
        delattr(self, 'tester')

    def update_dict(self, args):
        for k, v in vars(args).items():
            if v == '' or not hasattr(self, k): continue
            if k == 'gpus': setattr(self, 'gpus_str', v)
            setattr(self, k, self.transtype_a2b(v, getattr(self, k)))

    def setup_head(self, dataset):
        num_classes = dataset.num_classes
        heads = {'hm': num_classes, 'wh': 2 if not self.cat_spec_wh else 2 * num_classes}
        if self.reg_offset:
            heads.update({'reg': 2})
        if self.reg_obj:
            heads.update({'obj': 1})
        setattr(self, 'heads', heads)

    @staticmethod
    def transtype_a2b(a, b):
        if isinstance(b, (list, tuple)):
            out = a.split(',')
            for i in range(len(out)):
                out[i] = type(b[0])(out[i])
            return out
        if type(b) == type(True):
            a = int(a)
        return type(b)(a)

    def __repr__(self):
        string = 'config('
        for k, v in self.__dict__.items():
            if isinstance(v, dict): continue
            string += '{}: {}, '.format(k, v)
        string = string[:-2] + ')'
        return string


def get_cfg():
    from .defaults import _C
    _C.convert()
    return _C


def parse_args():
    parser = argparse.ArgumentParser(description='CenterNet')
    parser.add_argument('--model_cfg', required=True, help='')
    parser.add_argument('--data_dir', default='')
    parser.add_argument('--save_dir', default='')
    parser.add_argument('--id', default='')
    parser.add_argument('--num_epochs', default='')
    parser.add_argument('--batch_size', default='')
    parser.add_argument('--lr', default='')
    parser.add_argument('--input_w', default='')
    parser.add_argument('--input_h', default='')
    parser.add_argument('--gray', default='')
    parser.add_argument('--down_ratio', default='')
    parser.add_argument('--hflip', default='')
    parser.add_argument('--vflip', default='')
    parser.add_argument('--rotation', default='')
    parser.add_argument('--scale', default='')
    parser.add_argument('--shear', default='')
    parser.add_argument('--val_step', default='')
    parser.add_argument('--sample_size', default='')
    parser.add_argument('--reg_obj', default='')
    parser.add_argument('--keep_res', default='')
    parser.add_argument('--gpus', default='')
    parser.add_argument('--head_conv', default='')
    args = parser.parse_args()
    return args
