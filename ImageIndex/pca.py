#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
将高维特征进行pca降维
'''

import numpy as np
from sklearn.decomposition import IncrementalPCA
import pickle

prefix="../../data/"
input="abstract_features/feature.txt"
output = "abstract_features/feature_low.txt"
input_dim = 512
dim = 32
parameter_path = "parameters/pac.txt"

class pca:
    def __init__(self, dim=32, load=True):
        '''
        :param dim: 降维到几维
        :param load: 是否从本地加载参数
        '''
        if load:
            with open(parameter_path, "rb") as f:
                self.ipca = pickle.load(f)
        else:
            self.ipca = IncrementalPCA(n_components=dim)

    def train(self, input_file_name=prefix+input, output_file_name=prefix+output):
        print("start pca train")
        input_file = open(input_file_name, "r")
        train_list = []
        i = 0
        for line in input_file:
            str_list = line.strip(' \n').split(' ')
            if not len(str_list) == input_dim:
                continue
            float_list = [float(i) for i in str_list]
            train_list.append(float_list)
            # print("pca: {}\r".format(i), end='')
            #print("pca: {}\r".format(i))
            i += 1

        input_file.close()

        train_list = np.array(train_list)
        self.ipca.fit(train_list)
        final_list = self.ipca.transform(train_list)

        output_file = open(output_file_name, "w")
        for float_list in final_list:
            str_list = [str(j) for j in float_list]
            output_file.write(' '.join(str_list) + ' \n')
        output_file.close()
        print("finish pca train")
        with open(parameter_path, "wb") as f:
            pickle.dump(self.ipca, f)

    def transform(self, X):
        return self.ipca.transform(X)


if __name__ == '__main__':
    pca = pca(dim=32, load=False)
    pca.train()