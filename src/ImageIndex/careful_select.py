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

from ImageIndex.evaluate_similarity import *
from ImageIndex import get_prefix
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error

prefix = get_prefix()

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
    for c in candidates:
        candidate_feature_path = feature_path+c['image']+".delf"
        # Read features.
        try:
            candidate_locations, _, candidate_descriptors, _, _ = feature_io.ReadFromFile(candidate_feature_path)
            c['similarity']=evaluate_similarity(candidate_locations, candidate_descriptors, query_locations, query_descriptors)
        except:
            print("Error in feature io:",c)
            c["similarity"]=0
    sorted_candidates = sorted(candidates, key=lambda candidate:candidate['similarity'], reverse=True)
    #print(sorted_candidates)
    if len(sorted_candidates) > k:
        return sorted_candidates[0:k]
    else:
        return sorted_candidates

if __name__ == '__main__':
    candidate_images = ["2","3","4","5","6"]
    query_image = "1"
    print(careful_select(query_image, candidate_images, 4))
