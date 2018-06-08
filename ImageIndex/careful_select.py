#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
在调用前，请确保所有图片的详细特征已被提取到detailed_features
包括被查询的图片
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __init__ import *

import sys
from operator import attrgetter
from delf import feature_io
from ImageIndex.evaluate_similarity import *

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

prefix="../../data/"


# def careful_select(query_image, candidate_images, k=10):
#     '''
#     从候选图片中挑出和查询图片最为接近的k张图片(没有后缀名JPEG)
#     :param query_image: 查询图片的名字(没有后缀名JPEG)
#     :param candidate_images: 候选图片的名字列表
#     :param k:
#     :return: 最接近的k张图片的名字列表（按照相似度排序）
#     '''
#     candidates = [
#         {'image':image, 'similarity':0}
#         for image in candidate_images
#     ]
#     query_feature_path = "detailed_features/"+query_image+".delf"
#     query_locations, _, query_descriptors, _, _ = feature_io.ReadFromFile(query_feature_path)
#     print("query:",query_descriptors.shape)
#     for c in candidates:
#         candidate_feature_path = "detailed_features/"+c['image']+".delf"
#         # Read features.
#         candidate_locations, _, candidate_descriptors, _, _ = feature_io.ReadFromFile(candidate_feature_path)
#         c['similarity']=evaluate_similarity(query_locations, query_descriptors, candidate_locations, candidate_descriptors)
#         print(c["image"], candidate_descriptors.shape)
#     sorted_candidates = sorted(candidates, key=lambda candidate:candidate['similarity'], reverse=True)
#     if len(sorted_candidates) > k:
#         return sorted_candidates[0:k]
#     else:
#         return sorted_candidates

def careful_select(query_locations, query_descriptors, candidate_images, k=10, feature_path=prefix+"detailed_features/"):
    '''
    从候选图片中挑出和查询图片最为接近的k张图片(没有后缀名JPEG)
    :param query_locations: 查询图片的特征点位置
    :param query_descriptors: 查询图片的特征点值
    :param candidate_images: 候选图片的名字列表
    :param k:
    :return: 最接近的k张图片的名字列表（按照相似度排序）
    '''
    candidates = [
        {'image':image, 'similarity':0}
        for image in candidate_images
    ]
    print("query:",query_descriptors.shape)
    for c in candidates:
        candidate_feature_path = feature_path+c['image']+".delf"
        # Read features.
        candidate_locations, _, candidate_descriptors, _, _ = feature_io.ReadFromFile(candidate_feature_path)
        c['similarity']=evaluate_similarity(query_locations, query_descriptors, candidate_locations, candidate_descriptors)
        print(c["image"], candidate_descriptors.shape)
    sorted_candidates = sorted(candidates, key=lambda candidate:candidate['similarity'], reverse=True)
    if len(sorted_candidates) > k:
        return sorted_candidates[0:k]
    else:
        return sorted_candidates

if __name__ == '__main__':
    candidate_images = ["2","3","4","5","6"]
    query_image = "1"
    print(careful_select(query_image, candidate_images, 4))
