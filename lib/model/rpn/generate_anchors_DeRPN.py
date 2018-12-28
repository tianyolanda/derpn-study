from __future__ import print_function
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np
import pdb


try:
    xrange          # Python 2
except NameError:
    xrange = range  # Python 3

def generate_anchor_strings(base_size,w_an = np.array([16,32,64,128,256,512,1024])):
    # 生成 anchor string，也就是(x,w) pair
    x_ctr = np.array[base_size,base_size] - 1

    # anchors_strings = np.hstack((x_ctr - 0.5 * (ws - 1),x_ctr + 0.5 * (ws - 1)))
    anchors_strings = np.hstack((x_ctr - 0.5 * (w_an[i] - 1),x_ctr + 0.5 * (w_an[i] - 1))
                         for i in xrange(w_an.shape[0]))

    return anchors_strings


if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print(time.time() - t)
    print(a)
    from IPython import embed; embed()
