import numpy as np
import torch
# 测试版本的anchorstring_to_proposal_DeRPN.py，可运行！！！！！
# 输出的proposals格式：(xmin, ymin, xmax, ymax, scores)

# 22 是从anchorstring生成proposal，完成的是anchorstring_to_proposal_DeRPN.py 这个文件
# 目前出现的问题是，

# 参数设置
base_size = 16  # 60*40的feature map上生成anchor_string的起始点
combination_topk = 3  # 对每个(x,w)，逐像素寻找对应的(y,h). 每个(x,w)选取score前3的(y,h)
#  因为是以像素点为基准，所以，每个同样x(不同w)对应的前三名(y,h)是一样的！

# 初始化anchors_strings_w和anchors_strings_h
w_an = np.array([16,32,64,128,256,512,1024])
x_ctr = np.array([base_size]) - 1
ws = w_an[:, np.newaxis]
anchors_strings_w = np.hstack((np.round(x_ctr - 0.5 * (ws - 1)),  # x起始坐标
                             x_ctr + 0.5 * (ws - 1)))  # x终止坐标

anchors_strings_h = anchors_strings_w
anchors_strings_w = torch.from_numpy(anchors_strings_w)

anchors_strings_h = torch.from_numpy(anchors_strings_h)
anh_diejia = anchors_strings_w
for i in range(10):
    anh_diejia = anh_diejia + 10
    anchors_strings_h = torch.cat([anchors_strings_h, anh_diejia],0) # [77, 2]


# 把anchors_strings_w复制了3次，以便后续跟选出的(y,h)结合
a5 = np.tile(anchors_strings_w, combination_topk)
a5 = torch.from_numpy(a5).float()
a5 = a5.view(-1,2)  #  [3*7,2]


# 初始化order_w: 存储 每个anchors_strings 代表着16800中的第几个位置(anchor编号)
# t0就是原始的order_w [16000] 维度的这个
t0 = np.ones((7))
t0 = torch.from_numpy(t0)
_, order_w = torch.sort(t0)
order_w = (order_w)*7 + 2
order_w = order_w/7
# 这个在真实情况下是已有的order

# 初始化score_h：存储每个anchors_strings_h的score, 要根据这个score_h来找每个像素点中，符合要求的前3个（y,h）
score_h = np.arange(11*7)
score_h = torch.from_numpy(score_h)
score_h = score_h.view(-1,7)
_, order_h = torch.sort(score_h)

score_h = score_h.unsqueeze(2).float()

# 变成[11,7,2]，11代表是第几个像素点，便于后续匹配时按照像素点查找
anchors_strings_h = anchors_strings_h.view(-1,7,2).float() # ([11, 7, 2])
anchors_strings_h = torch.cat((anchors_strings_h,score_h),2)
# print(anchors_strings_h_score)


# 再每个像素点，根据score_h, 挑选出前combination_topk的order_h
if combination_topk > 0 and combination_topk < score_h.size(1):  # 防止数组越界
    order_h = order_h[:,:combination_topk]  # [11,3]

anchors_strings_h = anchors_strings_h[:,order_h]  # ([11,11, 3, 2])
h_temp = torch.empty([11,3,3]) # 这里的11要改成对应的像素点的数目

for i in range(anchors_strings_h.size(1)):
    h_temp[i] = anchors_strings_h[i,i,:,:]

anchors_strings_h = h_temp
# print(anchors_strings_h)

# 在每个像素点，挑选出前combination_topk的(y,h)
# anchors_strings_h = anchors_strings_h[:, order_h[i], :]  # ([11, 3, 2])

# 初始化一个tensor存储anchor_string_w依次对应的3个(y,h)，size和a5一样
new_anchors_strings_h = torch.zeros((a5.size(0),3), dtype=torch.float)

# 将new_anchors_strings_h赋值（需要使用循环来查找w对应的h）
for i in range(len(order_w)):
    new_anchors_strings_h[i*combination_topk:i*combination_topk+combination_topk,:] = anchors_strings_h[order_w[i],:,:].squeeze(0)

print(new_anchors_strings_h.size())
# 组合两个anchor_string

# 需要按照(左上角x坐标，左上角y坐标，右下角x坐标,右下角y坐标)来组成proposal
proposals_xmin = a5[:, 0]
proposals_xmax = a5[:, 1]
proposals_ymin = new_anchors_strings_h[:, 0]
proposals_ymax = new_anchors_strings_h[:, 1]
proposals_scores_h = new_anchors_strings_h[:, 2]
proposals_scores_w = proposals_scores_h
proposals_scores = 2 / ((1 / proposals_scores_h) + (1 / proposals_scores_w))
print(proposals_scores)

proposals_scores = proposals_scores.unsqueeze(1)
temp1 = torch.stack([proposals_xmin, proposals_ymin], 0).transpose(0, 1)
temp2 = torch.stack([proposals_xmax, proposals_ymax], 0).transpose(0, 1)
proposals = torch.cat((temp1, temp2), 1)
proposals = torch.cat((proposals, proposals_scores), 1)
print(proposals.size())


