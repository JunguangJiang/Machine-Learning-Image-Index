#!/usr/bin/python
# -*- coding: UTF-8 -*-
import os
import shutil
'''
验证集和测试集的划分
'''

train_ratio = 0.8 # 训练集的占比

if __name__ == '__main__':
    with open("imagelist.txt", 'r') as f:
        cateogory = {}
        for line in f:
            label = line.split('_')[0]
            if cateogory.get(label) == None:
                cateogory[label] = [line,]
            else:
                cateogory[label].append(line)

    train_file = open("train/imagelist.txt","w")
    validate_file = open("validate/imagelist.txt","w")
    for c in cateogory:
        if not os.path.exists("train/"+c):
            os.mkdir("train/"+c)
        if not os.path.exists("validate/"+c):
            os.mkdir("validate/"+c)
        if not os.path.exists("image/"+c):
            os.mkdir("image/"+c)
        list = cateogory[c]
        div = int(len(list)*train_ratio)
        for i in range(0, div):
            train_file.write(list[i])
            shutil.copy(list[i].strip('\n'), "train/"+c)
        for i in range(div, len(list)):
            validate_file.write(list[i])
            shutil.copy(list[i].strip('\n'), "validate/"+c)
        for i in range(0, len(list)):
            shutil.copy(list[i].strip('\n'), "image/" + c)
    train_file.close()
    validate_file.close()

