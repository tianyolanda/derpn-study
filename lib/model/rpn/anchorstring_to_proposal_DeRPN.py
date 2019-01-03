from __future__ import print_function


import numpy as np
import torch


def anchorstring_to_proposal(anchor_strings_w, anchor_strings_h, scores_w, scores_h,
                                               combination_topN, combination_topk, batch_size):
    # w,h 分别 16800个anchor string
    # anchor_strings_w: [batch_size,16800,2]
    # anchor_strings_h: [batch_size,16800,2]
    # combination_topN :
    m = 0
    _, order = torch.sort(scores_w, 1, True)  # 对第1维进行排序 ，True：降序
    output_w = scores_w.new(batch_size, combination_topN, 5).zero_() # 新建一个tensor，
    output_h = scores_h.new(batch_size, combination_topk, 5).zero_()
    for i in range(batch_size):
        scores_w_single = scores_w[i]
        anchor_strings_w_single = anchor_strings_w[i]
        if combination_topN > 0 and combination_topN < scores_w.numel():  # 防止数组越界
            order_single = order_single[:combination_topN]  # NMS前，留下12000个。 [12000]

    return m
