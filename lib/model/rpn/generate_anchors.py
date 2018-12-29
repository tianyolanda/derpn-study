from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])

try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3


def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    # scale 是 2的3，4，5次方 -- 128*128，256*256，512*512三种大小的anchor
    anchor.shape : [9,4]
    """

    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    # base_anchor 每个值依次减1 --> [0,0,15,15]
    # 基础anchor：[左上坐标(0,0)，右下坐标(15,15)] ，以它为基础，生成各种ratios的anchor
    ratio_anchors = _ratio_enum(base_anchor, ratios)  # []
    # 得到了一堆横向堆叠的基于比例的anchors
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    # np.vstack():在竖直方向上堆叠
    # 得到了一堆竖向堆叠的基于scale的anchor们（一个tensor）
    return anchors

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    返回输入anchor的长、宽和中心坐标x、y
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    因为每个anchor的存储是用 [左上xy坐标和右下xy坐标] 来存储的，一次你需要把w，h，xcenter，ycenter换算成这俩坐标
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),# 左上角x坐标
                         y_ctr - 0.5 * (hs - 1),# 左上角y坐标
                         x_ctr + 0.5 * (ws - 1),# 右下角x坐标
                         y_ctr + 0.5 * (hs - 1))) # 右下角y坐标
    # np.hstack():在水平方向上平铺, 就变成了
    # [x_ctr - 0.5 * (ws - 1)，y_ctr - 0.5 * (hs - 1)，x_ctr + 0.5 * (ws - 1)，y_ctr + 0.5 * (hs - 1)]
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    对于输入的anchor，列举它所有【横纵比】对应的anchor
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h # 128*128
    size_ratios = size / ratios # 128*128 /2 = 64 *128
    ws = np.round(np.sqrt(size_ratios)) # np.round() 浮点数x的四舍五入值，宽：64*根2
    hs = np.round(ws * ratios) # 长 128*根2
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr) # 制作anchor
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    对于输入的anchor，列举它所有【scale】对应的anchor
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
