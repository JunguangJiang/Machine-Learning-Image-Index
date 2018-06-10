#!/usr/bin/python
# -*- coding: UTF-8 -*-
# from __init__ import *
from ImageIndex.classify_model_feature import classify_model
# from ImageIndex.pca import pca
# from ImageIndex.extract_features import attention_model
# from ImageIndex.careful_select import careful_select
# from ImageIndex.data_search import search_tree

# import matplotlib.pyplot as plt
#from PIL import Image

from ImageIndex import get_prefix
prefix = get_prefix()
config_path = "delf_config_detailed.pbtxt"
image_list_path = "image/imagelist.txt"

feature_path={
    "label":"abstract_features/label_feature.txt",
    "feature":"abstract_features/feature.txt",
    "feature_low":"abstract_features/feature_low.txt"
}

dimension={
    "label":10,
    "feature":512,
    "feature_low":32
}


class image_index:
    '''
    图片检索系统的整合版本
    '''
    def __init__(self, type="feature"):
        self.type = type
        # self.classify_model = classify_model()
        # self.search_tree = search_tree(image_list_path=prefix+image_list_path, feature_path=prefix+feature_path[type])
        # self.pca = pca()
        #self.attention_model = attention_model(config_path)

    def filter_search(self, image_path, k=10):
        '''
        筛选图片
        :param image_path:图片路径名
        :param k: 返回多少张
        :return: 最像的10张图片,不含后缀
        '''
        pass
        # feature, label_feature = self.classify_model.get_single_feature(image_path)
        # if self.type == "label":
        #     query_feature = label_feature
        # elif self.type == "feature":
        #     query_feature = feature
        # else:
        #     query_feature = self.pca.transform(feature).tolist()
        # similar_list = self.search_tree.query(query_feature,k=k)
        # similar_list = [s.strip('.JPEG') for s in similar_list]
        # return similar_list

    def careful_search(self,image_path, k=10, multiple=4):
        '''
        精挑图片
        :param image_path: 图片路径名
        :param k: 返回多少张
        :param multiple:候选者的倍数
        :return: 最像的10张图片,不含后缀;以及相似度的平均分数
        '''
        # similar_list = self.filter_search(image_path,int(k*multiple))
        # locations_out, descriptors_out, feature_scales_out, attention_out = self.attention_model.get_single_attention(image_path)
        # similar_list = careful_select(locations_out, descriptors_out, similar_list, k)
        # average_similarity = self.count_average_similarity(similar_list)
        # similar_list = [s["image"] for s in similar_list]
        # average_similarity=0
        # return similar_list,average_similarity
        pass

    def count_average_similarity(self, similar_list):
        '''计算列表中图片相似度值的平均分，用于实验2'''
        total=0.0
        for s in similar_list:
            total += s["similarity"]
        total /= len(similar_list)
        return total

# def show_file(image_list, image_folder=prefix+"image/"):
#     '''
#     将若干张图片显示
#     :param image_list: 图片列表
#     :return:
#     '''
#     i = 1
#     plt.figure(figsize=(10, 5))
#     for l in image_list:
#         image_path = image_folder + l + ".JPEG"
#         image = Image.open(image_path)
#         plt.subplot(2, 5, i)
#         plt.title(l)
#         i += 1
#         plt.imshow(image)
#         plt.xticks([])
#         plt.yticks([])
#         plt.axis('off')
#     plt.show()
#
#
# def save_file(image_list, save_file=None, image_folder=prefix+"image/"):
#     '''
#     将若干张图片一起存入到本地
#     :param image_list: 图片列表
#     :param save_file: 保存的路径
#     :return:
#     '''
#     i=1
#     plt.figure(figsize=(10, 5))
#     for l in image_list:
#         image_path = image_folder+l+".JPEG"
#         image = Image.open(image_path)
#         plt.subplot(2,5,i)
#         plt.title(l)
#         #plt.figure(l["image"])
#         i+=1
#         plt.imshow(image)
#         plt.xticks([])
#         plt.yticks([])
#         plt.axis('off')
#     if save_file:
#         plt.savefig(save_file)


'''进行测试的图片名'''
test_image=[
    "n11669921_12332.JPEG",
    "n11669921_43145.JPEG",
    "n10247358_14658.JPEG",
    "n07897438_4679.JPEG",
    "n07897438_1733.JPEG",
    "n04583620_4028.JPEG",
    "n04515003_37361.JPEG",
    "n04515003_16807.JPEG",
    "n03877845_5487.JPEG",
    "n03877845_5041.JPEG",
    "n03767203_3741.JPEG",
    "n02278980_5577.JPEG",
    "n01613177_1805.JPEG",
    "n01923025_3201.JPEG"
]

# def test_feature_and_model(feature_type="feature", model="careful"):
#     '''
#     采用网络model，提取特征feature_type,并将10近邻结果存到本地
#     :param feature_type:
#     :param model:
#     :return:
#     '''
#     image_index_system = image_index(feature_type)
#     suffix=str(dimension[feature_type])+model[0]
#     if model=="careful":
#         for image in test_image:
#             similar_list = image_index_system.careful_search(image_path=prefix + "image/" + image)
#             image_index_system.save_file(similar_list, prefix + "result/" + image.strip(".JPEG") + "_" + suffix + ".pdf")
#     else:
#         for image in test_image:
#             similar_list = image_index_system.filter_search(image_path=prefix + "image/" + image)
#             save_file(similar_list, prefix + "result/" + image.strip(".JPEG") + "_" + suffix + ".pdf")
#
#
# def test_candidate_size():
#     '''
#     未完成
#     候选者选为查询结果的multiple倍，检测查询结果的相似度
#     '''
#     image_index_system = image_index()
#     print("不同multiple时的查询结果相似度测试")
#     with open(prefix+"result/test_candidate_size.csv","w") as f:
#         for image in test_image:
#             f.write(image+",")
#             for multiple in [1.5,2,2.5,3,3.5,4,4.5,5]:
#                 similar_list, average_similarity = image_index_system.careful_search(
#                     image_path=prefix+"image/"+image, multiple=multiple)
#                 f.write(str(average_similarity)+",")
#             f.write("\n")



# def batch_query(input_file, output_file, output_image_folder):
#     '''
#     未完成
#     进行批量查询
#     :param input_file: 查询文件
#     :param output_file: 结果文件
#     :param output_image_folder: 输出的图片文件夹
#     '''
#     query_image_list=[]
#     with open(input_file) as f:
#         for each_line in f:
#             query_image_list.append(each_line.strip('\n'))
#
#     with open(output_file) as f:
#         image_index_system = image_index()
#         for image in query_image_list:
#             similar_list, average_similarity = image_index_system.careful_search(mage_path=prefix + "image/" + image)
#             f.write(image+":"+",".join(similar_list))
#             if output_image_folder:
#                 save_file(similar_list, prefix + "result/" + image.strip(".JPEG") + ".pdf")


if __name__ == '__main__':
    # 实验1
    # for feature_type in ["feature","feature_low", "label"]:
    #     for model in ["careful", "filter"]:
    #         test_feature_and_model(feature_type, model)
    # 实验2
    # test_candidate_size()
    image_index_system = image_index()



