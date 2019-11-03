from .config import Cfg

_C = Cfg()

_C.model['down_ratio'] = 4

_C.datasets['data_dir'] = '/home/amax/qszn/ciwa_train_data'
_C.datasets['gray'] = False
_C.datasets['hflip'] = 0.5
_C.datasets['vflip'] = 0.5
_C.datasets['rotation'] = 180
_C.datasets['scale'] = 0.2
_C.datasets['shear'] = 0.

_C.trainer['gpus'] = (4, 5, 6)
_C.trainer['gpus_str'] = '4,5,6'
_C.trainer['lr'] = 1.25e-4
_C.trainer['lr_step'] = (90, 120)
_C.trainer['num_epochs'] = 140
_C.trainer['batch_size'] = 32
_C.trainer['num_workers'] = 16
_C.trainer['resume'] = False
_C.trainer['trainval'] = False
_C.trainer['input_w'] = 416
_C.trainer['input_h'] = 416
_C.trainer['down_ratio'] = _C.model['down_ratio']
_C.trainer['cat_spec_wh'] = False
_C.trainer['reg_offset'] = True
_C.trainer['reg_obj'] = False
_C.trainer['mse_loss'] = False
_C.trainer['reg_loss'] = 'l1'
_C.trainer['norm_wh'] = False
_C.trainer['dense_wh'] = False
_C.trainer['eval_oracle_hm'] = False
_C.trainer['eval_oracle_wh'] = False
_C.trainer['eval_oracle_offset'] = False
_C.trainer['num_stacks'] = 1
_C.trainer['hm_weight'] = 1
_C.trainer['wh_weight'] = 0.1
_C.trainer['off_weight'] = 1
_C.trainer['debug'] = 0
_C.trainer['val_step'] = 5
_C.trainer['sample_size'] = 8

_C.tester['keep_res'] = True
_C.tester['trainval'] = True
_C.tester['K'] = 100
_C.tester['vis_thresh'] = 0.1
