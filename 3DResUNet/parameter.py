size = 48  # 使用48张连续切片作为网络的输入

down_scale = 0.5  # 横断面降采样因子

expand_slice = 20  # 仅使用包含肝脏以及肝脏上下20张切片作为训练样本

slice_thickness = 1  # 将所有数据在z轴的spacing归一化到1mm

upper, lower = 200, -200  # CT数据灰度截断窗口

drop_rate = 0.3  # dropout随机丢弃概率

gpu = '0'  # 使用的显卡序号

Epoch = 100

learning_rate = 1e-4

learning_rate_decay = [500, 750]

alpha = 0.33  # 深度监督衰减系数

batch_size = 1

num_workers = 3

pin_memory = True

cudnn_benchmark = True

threshold = 0.5  # 阈值度阈值

stride = 12  # 滑动取样步长

maximum_hole = 5e4  # 最大的空洞面积

