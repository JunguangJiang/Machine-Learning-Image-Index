# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Extracts DELF features from a list of images, saving them to file.

The images must be in JPG format. The program checks if descriptors already
exist, and skips computation for those.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __init__ import *

import argparse
import os
import time


import tensorflow as tf

from google.protobuf import text_format
from delf import delf_config_pb2
from delf import feature_extractor
from delf import feature_io

cmd_args = None

# Extension of feature files.
_DELF_EXT = '.delf'

# Pace to report extraction log.
_STATUS_CHECK_ITERATIONS = 100


def _ReadImageList(list_path):
  """Helper function to read image paths.

  Args:
    list_path: Path to list of images, one image path per line.

  Returns:
    image_paths: List of image paths.
  """
  with tf.gfile.GFile(list_path, 'r') as f:
    image_paths = f.readlines()
  image_paths = [entry.rstrip() for entry in image_paths]
  return image_paths


class attention_model:
    def __init__(self, config_path):
        tf.logging.set_verbosity(tf.logging.INFO)

        # Parse DelfConfig proto.
        self.config = delf_config_pb2.DelfConfig()
        with tf.gfile.FastGFile(config_path, 'r') as f:
            text_format.Merge(f.read(), self.config)

        # Tell TensorFlow that the model will be built into the default Graph.
        with tf.Graph().as_default():
            self.sess = tf.Session()
            # Initialize variables.
            init_op = tf.global_variables_initializer()
            self.sess.run(init_op)

            # Loading model that will be used.
            tf.saved_model.loader.load(self.sess, [tf.saved_model.tag_constants.SERVING],
                                       self.config.model_path)
            self.graph = tf.get_default_graph()

    def get_attention(self, list_images_path, output_dir):
        '''
        从一组图片中提取出关键点特征到文件中
        :param list_images_path: 图片列表的存储路径
        :param output_dir: 输出文件夹
        :return:
        '''
        # Read list of images.
        tf.logging.info('Reading list of images...')
        image_paths = _ReadImageList(list_images_path)
        num_images = len(image_paths)
        tf.logging.info('done! Found %d images', num_images)

        # Create output directory if necessary.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        with self.graph.as_default():
            # Reading list of images.
            filename_queue = tf.train.string_input_producer(image_paths, shuffle=False)
            reader = tf.WholeFileReader()
            _, value = reader.read(filename_queue)
            image_tf = tf.image.decode_jpeg(value, channels=3)

            input_image = self.graph.get_tensor_by_name('input_image:0')
            input_score_threshold = self.graph.get_tensor_by_name('input_abs_thres:0')
            input_image_scales = self.graph.get_tensor_by_name('input_scales:0')
            input_max_feature_num = self.graph.get_tensor_by_name(
                'input_max_feature_num:0')
            boxes = self.graph.get_tensor_by_name('boxes:0')
            raw_descriptors = self.graph.get_tensor_by_name('features:0')
            feature_scales = self.graph.get_tensor_by_name('scales:0')
            attention_with_extra_dim = self.graph.get_tensor_by_name('scores:0')
            attention = tf.reshape(attention_with_extra_dim,
                                   [tf.shape(attention_with_extra_dim)[0]])

            locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
                boxes, raw_descriptors, self.config)

            # Start input enqueue threads.
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            start = time.clock()
            for i in range(num_images):
                # Write to log-info once in a while.
                if i == 0:
                    tf.logging.info('Starting to extract DELF features from images...')
                elif i % _STATUS_CHECK_ITERATIONS == 0:
                    elapsed = (time.clock() - start)
                    tf.logging.info('Processing image %d out of %d, last %d '
                                    'images took %f seconds', i, num_images,
                                    _STATUS_CHECK_ITERATIONS, elapsed)
                    start = time.clock()

                # # Get next image.
                im = self.sess.run(image_tf)

                # If descriptor already exists, skip its computation.
                out_desc_filename = os.path.splitext(os.path.basename(
                    image_paths[i]))[0] + _DELF_EXT
                out_desc_fullpath = os.path.join(cmd_args.output_dir, out_desc_filename)
                if tf.gfile.Exists(out_desc_fullpath):
                    tf.logging.info('Skipping %s', image_paths[i])
                    continue

                # Extract and save features.
                (locations_out, descriptors_out, feature_scales_out,
                 attention_out) = self.sess.run(
                    [locations, descriptors, feature_scales, attention],
                    feed_dict={
                        input_image:
                            im,
                        input_score_threshold:
                            self.config.delf_local_config.score_threshold,
                        input_image_scales:
                            list(self.config.image_scales),
                        input_max_feature_num:
                            self.config.delf_local_config.max_feature_num
                    })

                feature_io.WriteToFile(out_desc_fullpath, locations_out,
                                       feature_scales_out, descriptors_out,
                                       attention_out)

            # Finalize enqueue threads.
            coord.request_stop()
            coord.join(threads)

    def get_single_attention(self, image_file):
        '''
        从一张图片中提取关键点特征向量
        :param image_file: 图片名
        :return: 关键点的位置和对应的特征向量
        '''
        with self.graph.as_default():
            image_raw_data = tf.gfile.FastGFile(image_file, 'rb').read()
            image_tf = tf.image.decode_jpeg(image_raw_data, channels=3)

            input_image = self.graph.get_tensor_by_name('input_image:0')
            input_score_threshold = self.graph.get_tensor_by_name('input_abs_thres:0')
            input_image_scales = self.graph.get_tensor_by_name('input_scales:0')
            input_max_feature_num = self.graph.get_tensor_by_name(
                'input_max_feature_num:0')
            boxes = self.graph.get_tensor_by_name('boxes:0')
            raw_descriptors = self.graph.get_tensor_by_name('features:0')
            feature_scales = self.graph.get_tensor_by_name('scales:0')
            attention_with_extra_dim = self.graph.get_tensor_by_name('scores:0')
            attention = tf.reshape(attention_with_extra_dim,
                                   [tf.shape(attention_with_extra_dim)[0]])

            locations, descriptors = feature_extractor.DelfFeaturePostProcessing(
                boxes, raw_descriptors, self.config)


            im = self.sess.run(image_tf)

            # Extract and save features.
            (locations_out, descriptors_out, feature_scales_out,
             attention_out) = self.sess.run(
                [locations, descriptors, feature_scales, attention],
                feed_dict={
                    input_image:
                        im,
                    input_score_threshold:
                        self.config.delf_local_config.score_threshold,
                    input_image_scales:
                        list(self.config.image_scales),
                    input_max_feature_num:
                        self.config.delf_local_config.max_feature_num
                })
        return locations_out, descriptors_out, feature_scales_out, attention_out


    def __del__(self):
        if self.sess:
            self.sess.close()

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.register('type', 'bool', lambda v: v.lower() == 'true')
  parser.add_argument(
      '--config_path',
      type=str,
      default='delf_config_example.pbtxt',
      help="""
      Path to DelfConfig proto text file with configuration to be used for DELF
      extraction.
      """)
  parser.add_argument(
      '--list_images_path',
      type=str,
      default='list_images.txt',
      help="""
      Path to list of images whose DELF features will be extracted.
      """)
  parser.add_argument(
      '--output_dir',
      type=str,
      default='test_features',
      help="""
      Directory where DELF features will be written to. Each image's features
      will be written to a file with same name, and extension replaced by .delf.
      """)
  cmd_args, unparsed = parser.parse_known_args()
  model = attention_model(cmd_args.config_path)
  model.get_attention(cmd_args.list_images_path, cmd_args.output_dir)
  #locations_out, descriptors_out, feature_scales_out, attention_out = model.get_single_attention("../../data/image/n01613177_3.JPEG")