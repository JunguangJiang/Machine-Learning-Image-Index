#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
将高维特征进行pca降维
'''

import numpy as np
from sklearn.decomposition import IncrementalPCA

#input = "abstract_features/feature.txt"
input="image/feature.txt"
output = "abstract_features/feature_low.txt"
input_dim = 512
dim = 32

if __name__ == '__main__':
    input_file = open(input,"r")
    train_list=[]
    i = 0
    for line in input_file:
        str_list = line.strip(' \n').split(' ')
        if not len(str_list) == input_dim:
            continue
        float_list = [float(i) for i in str_list]
        train_list.append(float_list)
        print("pca: {}\r".format(i), end='')
    input_file.close()

    train_list = np.array(train_list)
    ipca = IncrementalPCA(n_components=dim)
    ipca.fit(train_list)
    final_list = ipca.transform(train_list)

    output_file = open(output, "w")
    for float_list in final_list:
        str_list = [str(j) for j in float_list]
        output_file.write(' '.join(str_list)+'\n')
    output_file.close()

