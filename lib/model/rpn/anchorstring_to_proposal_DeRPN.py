from __future__ import print_function


import numpy as np
import torch


def anchorstring_to_proposal(anchor_strings_w, anchor_strings_h, scores_w, scores_h,
                                               combination_topN, combination_topk, batch_size, _num_anchor_strings_w):
    # w,h 分别 16800个anchor string
    # anchor_strings_w: [batch_size,16800,2]
    # anchor_strings_h: [batch_size,16800,2]
    # scores_w, scores_h : [batch_size, 16800]
    # _num_anchor_strings_w : 每个point，anchor string定义数量：7

    # feature map 像素点总数
    fm_point_num = int(anchor_strings_w.size(1) / _num_anchor_strings_w)

    # combination_topN :
    _, order_w = torch.sort(scores_w, 1, True)  # 对第1维进行排序 ，True：降序

    output = anchor_strings_w.new(batch_size, combination_topN*combination_topk, 5).zero_()
    # 5是 (xmin, ymin, xmax, ymax, scores)

    for i in range(batch_size):
        # 将本batch中的全部值取出来
        scores_w_single = scores_w[i]
        scores_h_single = scores_h[i]
        anchors_strings_w_single = anchor_strings_w[i]
        anchors_strings_h_single = anchor_strings_h[i]
        order_w_single = order_w[i]

        # 把w的score信息和(x,w)信息结合起来
        scores_w_single = scores_w_single.unsqueeze(2).float()
        anchors_strings_w_single = torch.cat((anchors_strings_w_single, scores_w_single), 2)

        # 选出前combination_topN 个 anchor_strings_w
        if combination_topN > 0 and combination_topN < scores_w_single.numel():  # 防止数组越界
            order_w_single = order_w_single[:combination_topN]  # NMS前，留下12000个。 [12000]

        anchors_strings_w_single = anchors_strings_w_single[:, order_w_single[i], :]

        # 将 order_w 变换成 像素点的编号 ： 由 0~16800 变到 0~2400
        order_w_single = order_w_single / _num_anchor_strings_w

        # 把每个anchor_w复制combination_topk次，以便后续跟选出的(y,h)结合
        a5 = np.tile(anchors_strings_w_single, combination_topk)
        a5 = torch.from_numpy(a5).float()
        a5 = a5.view(-1, 3)  # [combination_topk*7,3]

        # 将 anchor_strings_h 的 score，按照每像素进行排序：每像素的7个h 排出0-7名
        scores_h_single = scores_h_single.view(-1, _num_anchor_strings_w)
        _, order_h_single = torch.sort(scores_h_single)
        scores_h_single = scores_h_single.unsqueeze(2).float()

        # 将anchors_strings_h的size从[16800]变成[2400,7,2]，其中2400表示feature map像素点个数，7表示每个像素点由7个h，2表示（y,h）
        # 2400代表是第几个像素点，便于后续匹配时按照像素点查找
        anchors_strings_h_single = anchors_strings_h_single.view(-1, _num_anchor_strings_w, 2).float()  # ([240, 7, 2])

        # 把h的score信息和(y,h)信息结合起来
        anchors_strings_h_single = torch.cat((anchors_strings_h_single, scores_h_single), 2)

        # 再每个像素点，根据score_h, 挑选出前combination_topk的order_h
        if combination_topk > 0 and combination_topk < scores_h_single.size(1):  # 防止数组越界
            order_h_single = order_h_single[:, :combination_topk]  # [2400,3]

        # 在每个像素点，挑选出前combination_topk个(y,h)
        anchors_strings_h_single = anchors_strings_h_single[:, order_h_single]
        h_temp = torch.empty([fm_point_num, combination_topk, 2])
        for i in range(anchors_strings_h_single.size(1)):
            h_temp[i] = anchors_strings_h_single[i, i, :, :]

        anchors_strings_h_single = h_temp

        # 初始化一个tensor存储anchor_string_w依次对应的3个(y,h)，size和a5一样
        new_anchors_strings_h_single = torch.empty_like(a5, dtype=torch.float)
        for i in range(len(order_w_single)):
            new_anchors_strings_h_single[i * combination_topk:i * combination_topk + combination_topk,
            :] = anchors_strings_h_single[order_w_single[i], :, :].squeeze(0)

        # 组合两个anchor_string

        # 需要按照(左上角x坐标，左上角y坐标，右下角x坐标,右下角y坐标)来组成proposal
        proposals_xmin = a5[:, 0]
        proposals_xmax = a5[:, 1]
        proposals_ymin = new_anchors_strings_h_single[:, 0]
        proposals_ymax = new_anchors_strings_h_single[:, 1]

        # 整合w,h的分数
        proposals_scores_w = a5[:, 2]
        proposals_scores_h = new_anchors_strings_h_single[:, 2]
        proposals_scores = 2 / ((1 / proposals_scores_h) + (1 / proposals_scores_w))
        proposals_scores = proposals_scores.unsqueeze(1)

        # 合并成一个proposal(xmin, ymin, xmax, ymax, scores)
        temp1 = torch.stack([proposals_xmin, proposals_ymin], 0).transpose(0, 1)
        temp2 = torch.stack([proposals_xmax, proposals_ymax], 0).transpose(0, 1)
        proposals = torch.cat((temp1, temp2), 1)
        proposals = torch.cat((proposals, proposals_scores), 1)

        # batch中的每个图都保存到output中
        output[i] = proposals

    return output
# output: [batch_size,combination_topN*combination_topk, 5]
