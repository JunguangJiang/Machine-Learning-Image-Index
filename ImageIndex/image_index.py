#!/usr/bin/python
# -*- coding: UTF-8 -*-
from __init__ import *
from ImageIndex.classify_model_feature import classify_model
from ImageIndex.pca import pca
from ImageIndex.extract_features import attention_model
from ImageIndex.careful_select import careful_select
from ImageIndex.data_search import search_tree

import matplotlib.pyplot as plt
from PIL import Image

prefix="../../data/"
config_path = "delf_config_detailed.pbtxt"
image_list_path = "image/imagelist.txt"
# feature_path = "abstract_features/label_feature.txt"

feature_path={
    "label":"abstract_features/label_feature.txt",
    "feature":"abstract_features/feature.txt",
    "feature_low":"abstract_features/feature_low.txt"
}


class image_index:
    '''
    图片检索系统的整合版本:只用分类网络
    '''
    def __init__(self, type="label"):
        self.type = type
        self.classify_model = classify_model()
        self.search_tree = search_tree(image_list_path=prefix+image_list_path, feature_path=prefix+feature_path[type])
        self.pca = pca()
        self.attention_model = attention_model(config_path)

    def filter_search(self, image_path, k=10):
        '''
        筛选图片
        :param image_path:图片路径名
        :return: 最像的10张图片
        '''
        feature, label_feature = self.classify_model.get_single_feature(image_path)
        if self.type == "label":
            query_feature = label_feature
        elif self.type == "feature":
            query_feature = feature
        else:
            query_feature = self.pca.transform(feature).tolist()
        similar_list = self.search_tree.query(query_feature,k=k)
        return similar_list

    def careful_search(self,image_path, k=10):
        similar_list = self.filter_search(image_path,k*4)
        similar_list = [s.strip('.JPEG') for s in similar_list]
        locations_out, descriptors_out, feature_scales_out, attention_out = self.attention_model.get_single_attention(image_path)
        similar_list = careful_select(locations_out, descriptors_out, similar_list, k)
        similar_list = [s["image"] for s in similar_list]
        return similar_list

    def show(self, image_list):
        i=1
        for l in image_list:
            image_path = prefix+"image/"+l+".JPEG"
            image = Image.open(image_path)
            plt.subplot(2,5,i)
            plt.title(l)
            #plt.figure(l["image"])
            i+=1
            plt.imshow(image)
        plt.show()



if __name__ == '__main__':
    p = image_index("feature")
    similar_list = p.careful_search(image_path=prefix+"image/n11669921_3845.JPEG")
    p.show(similar_list)
    #print(similar_list)
