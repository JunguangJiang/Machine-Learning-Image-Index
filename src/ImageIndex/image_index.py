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
from delf import feature_io

from ImageIndex import get_prefix
prefix = get_prefix()
suffix = ".JPEG"
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
        self.classify_model = classify_model()
        self.search_tree = search_tree(image_list_path=prefix+image_list_path, feature_path=prefix+feature_path[type])
        self.pca = pca()
        self.attention_model = attention_model(config_path)

    def filter_search(self, image_path, k=10):
        '''
        筛选图片
        :param image_path:图片路径名
        :param k: 返回多少张
        :return: 最像的10张图片,不含后缀
        '''
        # pass
        feature, label_feature = self.classify_model.get_single_feature(image_path)
        if self.type == "label":
            query_feature = label_feature
        elif self.type == "feature":
            query_feature = feature
        else:
            query_feature = self.pca.transform(feature).tolist()
        similar_list = self.search_tree.query(query_feature,k=k)
        similar_list = [s.strip(suffix) for s in similar_list]
        return similar_list

    def careful_search(self,image_path, k=10, multiple=4,DELFFile=None):
        '''
        精挑图片
        :param image_path: 图片路径名
        :param k: 返回多少张
        :param multiple:候选者的倍数
        :return: 最像的10张图片,不含后缀;以及相似度的平均分数
        '''
        similar_list = self.filter_search(image_path,int(k*multiple))
        if DELFFile:
            candidate_feature_path = DELFFile
            # Read features.
            try:
                locations_out, _, descriptors_out, _, _ = feature_io.ReadFromFile(candidate_feature_path)
            except:
                return similar_list[0:10],0
        else:
            locations_out, descriptors_out, feature_scales_out, attention_out = self.attention_model.get_single_attention(image_path)
        similar_list_new = careful_select(locations_out, descriptors_out, similar_list, k)
        average_similarity = self.count_average_similarity(similar_list_new)
        # if similar_list_new[9]['similarity'] > 20: #注意力网络需要比较图像有较多的相似度
        #     similar_list = similar_list_new
        #     similar_list = [s["image"] for s in similar_list]
        # else:
        #     similar_list = similar_list[0:k]

        similar_list_final = []
        for s in similar_list_new:
            if s['similarity'] > 20:
                similar_list_final.append(s['image'])
            if len(similar_list_final) == 10:
                break

        for s in similar_list:
            if len(similar_list_final) >= 10:
                break
            if s not in similar_list_final:
                similar_list_final.append(s)

        print(similar_list_final)

        return similar_list_final,average_similarity


    def count_average_similarity(self, similar_list):
        '''计算列表中图片相似度值的平均分，用于实验2'''
        total=0.0
        for s in similar_list:
            if s == similar_list[0]:
                continue
            total += s["similarity"]
        total /= (len(similar_list)-1)
        return total

def show_file(image_list, image_folder=prefix+"image/"):
    '''
    将若干张图片显示
    :param image_list: 图片列表
    :return:
    '''
    i = 1
    plt.figure(figsize=(10, 5))
    for l in image_list:
        image_path = image_folder + l + suffix
        image = Image.open(image_path)
        plt.subplot(2, 5, i)
        plt.title(l)
        i += 1
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    plt.show()


def save_file(image_list, save_file=None, image_folder=prefix+"image/"):
    '''
    将若干张图片一起存入到本地
    :param image_list: 图片列表
    :param save_file: 保存的路径
    :return:
    '''
    i=1
    plt.figure(figsize=(10, 5))
    for l in image_list:
        image_path = image_folder+l+suffix
        image = Image.open(image_path)
        plt.subplot(2,5,i)
        plt.title(l)
        #plt.figure(l["image"])
        i+=1
        plt.imshow(image)
        plt.xticks([])
        plt.yticks([])
        plt.axis('off')
    if save_file:
        plt.savefig(save_file)


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

def test_feature_and_model(feature_type="feature", model="careful"):
    '''
    采用网络model，提取特征feature_type,并将10近邻结果存到本地
    :param feature_type:
    :param model:
    :return:
    '''
    image_index_system = image_index(feature_type)
    suffix=str(dimension[feature_type])+model[0]
    if model=="careful":
        for image in test_image:
            similar_list = image_index_system.careful_search(image_path=prefix + "image/" + image)
            image_index_system.save_file(similar_list, prefix + "result/" + image.strip(suffix) + "_" + suffix + ".pdf")
    else:
        for image in test_image:
            similar_list = image_index_system.filter_search(image_path=prefix + "image/" + image)
            save_file(similar_list, prefix + "result/" + image.strip(suffix) + "_" + suffix + ".pdf")


def test_candidate_size():
    '''
    候选者选为查询结果的multiple倍，检测查询结果的相似度
    '''
    image_index_system = image_index()
    print("不同multiple时的查询结果相似度测试")
    with open(prefix+"result/test_candidate_size.csv","w") as f:
        for image in test_image:
            f.write(image+",")
            for multiple in [1.5,2,2.5,3,3.5,4,4.5,5]:
                similar_list, average_similarity = image_index_system.careful_search(
                    image_path=prefix+"image/"+image, multiple=multiple)
                f.write(str(average_similarity)+",")
            f.write("\n")


def batch_query(input_file, output_file, output_image_folder):
    '''
    进行批量查询
    :param input_file: 查询文件
    :param output_file: 结果文件
    :param output_image_folder: 输出的图片文件夹
    '''
    query_image_list=[]
    with open(input_file) as f:
        for each_line in f:
            query_image_list.append(each_line.strip('\n'))

    with open(output_file) as f:
        image_index_system = image_index()
        for image in query_image_list:
            similar_list, average_similarity = image_index_system.careful_search(image_path=prefix + "image/" + image)
            f.write(image+":"+",".join(similar_list))
            if output_image_folder:
                save_file(similar_list, prefix + "result/" + image.strip(suffix) + ".pdf")

def batch_query_quick(input_file, output_file, output_image_folder=None):
    '''
    进行快速批量查询
    :param input_file: 查询文件
    :param output_file: 结果文件
    :param output_image_folder: 输出的图片文件夹
    '''
    query_image_list = []
    with open(input_file,"r") as f:
        for each_line in f:
            query_image_list.append(each_line.strip('\n'))

    with open(output_file, "r") as f:
        for each_line in f:
            image=each_line.split(":")[0]+suffix
            query_image_list.remove(image)
            print("pass",image)

    with open(output_file,"a") as f:
        image_index_system = image_index()
        frequency=0

        for image in query_image_list:
            DELFile = prefix+"detailed_features/"+image.strip(suffix)+".delf"
            similar_list, average_similarity = image_index_system.careful_search(image_path=prefix + "image/" + image,
                                                                                 multiple=3,
                                                                                 DELFFile=DELFile)
            # similar_list = image_index_system.filter_search(image_path=prefix + "image/" + image)
            f.write(image.strip(suffix) + ":" + ",".join(similar_list)+"\n")
            frequency+=1
            if frequency %10 == 0:
                f.flush()
                print("Finish", frequency, "pictures")

            if output_image_folder:
                save_file(similar_list, prefix + "result/" + image.strip(suffix) + ".pdf")

def sort_file(input_file, output_file):
    '''对结果文件中的内容进行排序'''
    result=[]
    with open(input_file,"r") as f:
        for each_line in f:
            result.append({"id":each_line.split(":")[0], "content":each_line})

    result=sorted(result, key=lambda r:r["id"])

    with open(output_file, "w") as f:
        for r in result:
            f.write(r["content"])

def save_by_file(input_file):
    with open(input_file, "r") as f:
        for each_line in f:
            list = each_line.strip("\n").split(":")[1].split(",")
            save_file(list, save_file=prefix+"result/"+each_line.split(":")[0]+".pdf")

if __name__ == '__main__':
    # 实验1
    # for feature_type in ["feature","feature_low", "label"]:
    #     for model in ["careful", "filter"]:
    #         test_feature_and_model(feature_type, model)
    # 实验2
    # test_candidate_size()
    # 批量查询
    # batch_query_quick(input_file=prefix+image_list_path,output_file=prefix+"result/result.txt", output_image_folder=prefix+"result")
    # input_file = "image/imagelist.txt"
    # output_file = "result/result_temp.txt"
    # batch_query_quick(input_file=prefix+input_file,output_file=prefix+output_file)

    #sort_file(prefix+"result/result_temp.txt", prefix+"result/result_final.txt")
    #save_by_file(prefix+"result/result.txt")



