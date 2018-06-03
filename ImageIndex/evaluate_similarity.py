#!/usr/bin/python
# -*- coding: UTF-8 -*-

'''
对两个图片的多个关键点进行相似度度量，计算两个图片的相似度

The matching is done using feature-based nearest-neighbor search, followed by
geometric verification using RANSAC.

The DELF features can be extracted using the extract_features.py script.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __init__ import *

import numpy as np
from scipy.spatial import cKDTree
from skimage.measure import ransac
from skimage.transform import AffineTransform
import tensorflow as tf

from delf import feature_io

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"]='3' # 只显示 Error
_DISTANCE_THRESHOLD = 0.8

def evaluate_similarity(locations_1, descriptors_1, locations_2, descriptors_2):
    tf.logging.set_verbosity(tf.logging.INFO)

    num_features_1 = locations_1.shape[0]
    num_features_2 = locations_2.shape[0]

    # Find nearest-neighbor matches using a KD tree.
    d1_tree = cKDTree(descriptors_1)
    _, indices = d1_tree.query(
        descriptors_2, distance_upper_bound=_DISTANCE_THRESHOLD)

    # Select feature locations for putative matches.
    locations_2_to_use = np.array([
        locations_2[i,]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])
    locations_1_to_use = np.array([
        locations_1[indices[i],]
        for i in range(num_features_2)
        if indices[i] != num_features_1
    ])

    if len(locations_1_to_use) < 3:
        return 3

    # Perform geometric verification using RANSAC.
    _, inliers = ransac(
        (locations_1_to_use, locations_2_to_use),
        AffineTransform,
        min_samples=3,
        residual_threshold=20,
        max_trials=1000
    )

    tf.logging.info('Found %d inliers' % sum(inliers))

    return sum(inliers)

if __name__ == '__main__':
    # features_1_path = "test_features/1.delf"
    # features_2_path = "test_features/2.delf"
    # # Read features.
    # locations_1, _, descriptors_1, _, _ = feature_io.ReadFromFile(
    #     features_1_path)
    # num_features_1 = locations_1.shape[0]
    # tf.logging.info("Loaded image 1's %d features" % num_features_1)
    # locations_2, _, descriptors_2, _, _ = feature_io.ReadFromFile(
    #     features_2_path)
    # num_features_2 = locations_2.shape[0]
    # tf.logging.info("Loaded image 2's %d features" % num_features_2)
    # evaluate_similarity(locations_1, descriptors_1, locations_2, descriptors_2)

    features_1_path = "test_features/1.delf"
    features_2_path = "test_features/2.delf"
    # Read features.
    locations_1, features_out_1, descriptors_1, attention_out_1, _ = feature_io.ReadFromFile(
        features_1_path)
    num_features_1 = locations_1.shape[0]
    tf.logging.info("Loaded image 1's %d features" % num_features_1)
    locations_2, features_out_2, descriptors_2, attention_out_2, _ = feature_io.ReadFromFile(
        features_2_path)
    num_features_2 = locations_2.shape[0]
    tf.logging.info("Loaded image 2's %d features" % num_features_2)
    # evaluate_similarity(locations_1, descriptors_1, locations_2, descriptors_2)
    print(locations_1.shape)
    print(features_out_1)
    print(descriptors_1.shape)
    print(attention_out_1)