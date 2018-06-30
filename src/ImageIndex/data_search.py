import numpy as np
from sklearn.neighbors import BallTree

#读取图片文件名列表
def get_file_list(imagelist):
    file_list = []
    with open(imagelist,'r') as ff:
       for line in ff.readlines():
           file_list.append(line.strip())
    return file_list

#读取特征列表
def load_feature(featurefile):
    feature_list = []
    with open(featurefile,'r') as f1:
        for line in f1:
            feature_list.append([float(tk) for tk in line.split(' ')[:-1]])
    return np.array(feature_list)


class search_tree:
    '''
    检索树，基于ball tree实现
    '''
    def __init__(self, image_list_path, feature_path):
        self.x = load_feature(feature_path)
        self.tree = BallTree(self.x)
        self.file_list = get_file_list(image_list_path)
        #self.y = labelize_image(self.file_list)

    def query(self, feature, k):
        dist, ind = self.tree.query(feature, k=k)
        similar_list = [self.file_list[i] for i in ind[0]]
        return similar_list

