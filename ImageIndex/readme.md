## 预处理
- 训练分类网络 在ImageIndex 中 python classify_model_train.py
- 基于分类网络对图像进行粗特征提取  在ImageIndex 中 python classify_model_feature.py
- 对512维特征进行预处理 python pca.py
- 对图像数据集中的所有图片进行关键点特征提取 sh extract.sh

## 实际使用
- 进行批量查询
- 运行图形界面

## 文件功能说明
- parameters文件夹存储了训练后网络的参数
- resnet.py 是pytorch库文件，我们对其进行了略微改动
- classify_model_feature.py 定义了图像分类网络
- classify_model_train.py 对图像分类网络进行训练
- pca.py 对分类网络得到的高维向量进行PCA降维
- data_search.py 实现了高维数据的检索树（基于ball树）
- extract_feature.py 实现了注意力网络
- delf_config_detailed.pbtxt是注意力网络的参数配置文件
- extract.sh是用注意力网络对数据库中的所有图片进行关键点提取
- evaluate_similarity.py 对两个图片进行相似度度量（方法是关键点的比对）
- image_index.py 是对图像检索系统的一个整合

