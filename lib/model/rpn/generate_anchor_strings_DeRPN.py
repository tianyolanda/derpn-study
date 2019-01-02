from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import torch
import pdb


# try:
#     xrange          # Python 2
# except NameError:
#     xrange = range  # Python 3

def generate_anchor_strings(base_size=16, w_an = np.array([16,32,64,128,256,512,1024])):
    # 对每一个点，以它x坐标为中心，生成其对应的7种不同w的anchor string，存储为[xmin,xmax]格式
    # 纵坐标同理
    # 生成 anchor string，也就是(x,w) pair--> (xmin,xmax), 其中xmax-xmin = 各种w_an
    x_ctr = np.array([base_size]) - 1
    ws = w_an[:, np.newaxis]
    anchors_strings = np.hstack((np.round(x_ctr - 0.5 * (ws - 1)),  # x起始坐标
                                 x_ctr + 0.5 * (ws - 1)))  # x终止坐标
    return(anchors_strings)


if __name__ == '__main__':
    import time
    t = time.time()
    # a = generate_anchor_strings(w_an = np.array([16,32,64,128,256,512,1024]))
    a = torch.from_numpy(generate_anchor_strings(w_an = np.array([16,32,64,128,256,512,1024]))).float()
    # print('00000',a.size())
    # print(time.time() - t)
    print(a.size())
    # from IPython import embed; embed()
# 输出结果：
# [[   8.    22.5]
#  [  -0.    30.5]
#  [ -16.    46.5]
#  [ -48.    78.5]
#  [-112.   142.5]
#  [-240.   270.5]
#  [-496.   526.5]]