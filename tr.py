

from __future__ import print_function


import os
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.decomposition import TruncatedSVD

import pre


def train(src,dest,pivot_num,pivot_min_st):

    x, y, x_valid, y_valid,inputs = pre.preproc(pivot_num,pivot_min_st,src,dest)
    pivot_mat = np.zeros((pivot_num, inputs))
    for i in range(pivot_num):
        clf = linear_model.SGDClassifier(loss="modified_huber")
        clf.fit(x,y[:,i])
        pivot_mat[i]=clf.coef_
    print("finish training")
    
    pivot_mat=pivot_mat.transpose()
    svd50 = TruncatedSVD(n_components=50)
    pivot_mat50=svd50.fit_transform(pivot_mat)
    svd100 = TruncatedSVD(n_components=100)
    pivot_mat100=svd100.fit_transform(pivot_mat)
    svd150 = TruncatedSVD(n_components=150)
    pivot_mat150=svd150.fit_transform(pivot_mat)
    print("finished svd")



    base_weight_str = src + "_to_" + dest + "/weights/w_" + src + "_" + dest 
    
    filename = weight_str + "_50"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.save(filename, pivot_mat50)

    filename = weight_str + "_100"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.save(filename, pivot_mat100)

    filename = weight_str + "_150"
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    np.save(filename, pivot_mat150)

