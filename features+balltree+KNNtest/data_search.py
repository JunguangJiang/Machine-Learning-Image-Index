import numpy as np
from sklearn import neighbors
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import BallTree
import matplotlib.pyplot as plt
import pickle

#读取图片文件名列表
def get_file_list(imagelist):
    file_list = []
    with open(imagelist,'r') as ff:
       for line in ff.readlines():
           file_list.append(line.strip())
    return file_list

#标签化图片
def labelize_image(filelist):
    #将图片分类（根据文件名）
    label_list = []
    file_kind_set = set([])
    for filename in file_list:
        file_kind_set.add(filename[0:9])
    file_kind_list = [i for i in file_kind_set]
    for filename in file_list:
        label_list.append(file_kind_list.index(filename[0:9]))
    #print('already classify file by name')
    return np.array(label_list)

#读取特征列表
def load_feature(featurefile):
    feature_list = []
    with open(featurefile,'r') as f1:
        for line in f1:
            feature_list.append([float(tk) for tk in line.split(' ')[:-1]])
    return np.array(feature_list)

#KNN分类算法测试，参数为特征向量，标签值，测试数据比例，建树方法
def KNN_test(x,y,ratio,method):
    #拆分训练数据与测试数据
    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = ratio)
    #训练KNN分类器
    clf = neighbors.KNeighborsClassifier(algorithm = method)
    clf.fit(x_train,y_train)
    #print('already train KNN')

    #测试结果的打印
    answer = clf.predict(x)
    print(method,end = ' ')
    print(ratio,end = ' ')
    print(np.mean(answer == y))

#构建球树
def setup_ball_tree(x,filename):
    #可调节参数leaf_size = ?
    tree = BallTree(x)
    with open(filename,'wb') as f:
        pickle.dump(tree,f)
    return tree

#读取球树
def read_ball_tree(filename):
    with open('ball_tree.txt','rb') as f:
        tree = pickle.load(f)
    return tree

#查找指定数目的相似图片(包括自身),返回一个图片的文件名列表和查询正确率
def query_image(filelist,filename,tree,x,y,num):
    position = filelist.index(filename)
    if position == 0:
        print('no such file in database')
        return []
    else:
        dist,ind = tree.query([x[position]], k = num)
        similar_list = [filelist[i] for i in ind[0]]
        true_label = y[filelist.index(filename)]
        a = 0
        for j in similar_list:
            if y[filelist.index(j)] == true_label:
                a += 1               
        return similar_list,a/num

###########################分隔符############################################    

#直接在这里改路径
file_list = get_file_list('imagelist.txt')
x = load_feature('label_feature.txt')
y = labelize_image(file_list)
balltree = setup_ball_tree(x,'ball_tree.txt')
balltree = read_ball_tree('ball_tree.txt')
similar_list,precision = query_image(file_list,'n03877845_1184.JPEG',balltree,x,y,10)
print(similar_list)
print(precision)

#参数列表为特征向量，标签，测试集占比，方法
#其中，方法可以使用'brute','kd_tree','ball_tree'三种
KNN_test(x,y,0.5,'ball_tree')

