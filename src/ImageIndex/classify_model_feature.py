#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
从分类网络中提取每张图片的特征
'''

from __future__ import print_function, division
# from __init__ import *

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
import os
import resnet as rn
from PIL import Image
from __init__ import get_prefix

model_path = "parameters/classify_model.pt"#存储分类网络

prefix = get_prefix()
image_list = prefix+"image/imagelist_new.txt"#图片的路径
feature=prefix+"abstract_features/feature.txt" # 提取出的512维特征存储的文件
label_feature=prefix+"abstract_features/label_feature.txt"#10维特征存储的文件

data_transfroms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class classify_model:
    '''分类网络'''
    def __init__(self):
        if os.path.exists(model_path):
            self.model_conv = torch.load(model_path)
            print("exist model",model_path)
        else:
            self.model_conv = rn.resnet18(pretrained=True)

            for param in self.model_conv.parameters():
                param.requires_grad = False

            # Parameters of newly constructed modules have requires_grad=True by default
            num_ftrs = self.model_conv.fc.in_features
            self.model_conv.fc = nn.Linear(num_ftrs, 10)

            self.model_conv = self.model_conv.to(device)

    def get_single_feature(self, image_file):
        '''
        求一张图片的特征向量
        :param image_file: 图片存储路径
        :return: 图片的两个特征向量
        '''
        image = Image.open(image_file)
        sample = data_transfroms(image)
        sample = sample.view(-1, 3, 224, 224)
        outputs = self.model_conv(sample)

        feature_list = self.model_conv.feature.tolist()
        label_feature_list = outputs.tolist()
        return feature_list, label_feature_list


    def get_feature(self, image_list, feature, label_feature):
        '''
        求一组图片的特征向量
        :param image_list: 图片列表的存储文件名
        :param feature: 提取特征的存储路径
        :param label_feature: 提取标签特征的存储路径
        :return:
        '''
        image_list_file = open(image_list, "r")
        feature_file = open(feature, "w")
        label_feature_file = open(label_feature, "w")

        i = 0
        for line in image_list_file:
            image = Image.open(prefix+line.strip('\n'))
            sample = data_transfroms(image)
            sample = sample.view(-1, 3, 224, 224)
            outputs = self.model_conv(sample)

            list = self.model_conv.feature.tolist()
            for l in list[0]:
                feature_file.write(str(l) + " ")
            feature_file.write('\n')

            list = outputs.tolist()
            for l in list[0]:
                label_feature_file.write(str(l) + " ")
            label_feature_file.write('\n')

            i += 1
            if i % 100 == 0:
                feature_file.buffer.flush()
                label_feature_file.buffer.flush()
                print("extract feature {}/{}".format(i, 5613))

        image_list_file.close()
        feature_file.close()
        label_feature_file.close()

if __name__ == '__main__':
    model = classify_model()
    model.get_feature(image_list=image_list, feature=feature, label_feature=label_feature)



