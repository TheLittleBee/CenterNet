from .config import Cfg

_C = Cfg()

# 默认降采样设置为4,支持[2,4,8,16]
_C.model['down_ratio'] = 4

# 数据集相关参数配置
# 数据集所在路劲,绝对路径
_C.datasets['data_dir'] = '/home/amax/qszn/ciwa_train_data'
# 是否为灰度图像,默认为False,暂不支持灰度图像
_C.datasets['gray'] = False
# 水平翻转概率
_C.datasets['hflip'] = 0.5
# 垂直翻转概率
_C.datasets['vflip'] = 0.5
# 旋转角度
_C.datasets['rotation'] = 180
# 缩放尺度
_C.datasets['scale'] = 0.2
# 弹性形变
_C.datasets['shear'] = 0.

# 训练相关参数配置
# gpu配置
_C.trainer['gpus'] = (4, 5, 6)
# 学习率
_C.trainer['lr'] = 1.875e-4
# 学习率下降,默认乘0.1
_C.trainer['lr_step'] = (180, 210)
# 迭代次数
_C.trainer['num_epochs'] = 230
_C.trainer['batch_size'] = 48
_C.trainer['num_workers'] = 16
_C.trainer['resume'] = False
# 模型输入大小
_C.trainer['input_w'] = 416
_C.trainer['input_h'] = 416
# 若每个类别分别回归wh,则设为True
_C.trainer['cat_spec_wh'] = False
# 是否要回归中心点偏移量
_C.trainer['reg_offset'] = True
# 是否训练前景分类
_C.trainer['reg_obj'] = False
# heatmap用mse还是focal loss
_C.trainer['mse_loss'] = False
# 回归的loss类型,支持'l1'|'sl1'
_C.trainer['reg_loss'] = 'l1'
# 各部分loss的权重
_C.trainer['hm_weight'] = 1
_C.trainer['wh_weight'] = 0.1
_C.trainer['off_weight'] = 1
_C.trainer['obj_weight'] = 1
# valid数据路径
_C.trainer['val_dir'] = ''
# valid步数
_C.trainer['val_step'] = 100
# 可视化图片数
_C.trainer['sample_size'] = 8
# 以下为centernet默认参数,没有使用
_C.trainer['norm_wh'] = False
_C.trainer['dense_wh'] = False
_C.trainer['eval_oracle_hm'] = False
_C.trainer['eval_oracle_wh'] = False
_C.trainer['eval_oracle_offset'] = False
_C.trainer['num_stacks'] = 1
_C.trainer['debug'] = 0

# 测试相关参数配置
# 是否保持原图分辨率
_C.tester['keep_res'] = True
# 取topk结果
_C.tester['K'] = 100
# 可视化置信度阈值
_C.tester['vis_thresh'] = 0.1
# 保证分辨率能被32整除
_C.tester['pad'] = 31
# 在test数据集测试,还是valid数据集
_C.tester['trainval'] = True
