## 设计流程
- 训练分类网络 在ImageIndex 中 python classify_model_train.py
- 基于分类网络对图像进行粗特征提取  在ImageIndex 中 python classify_model_feature.py
- 对512维特征进行预处理 python pca.py
- 基于上述特征进行检索
- 对图像数据集中的所有图片进行关键点特征提取 sh extract.sh
- 对检索到的候选者进行验证，选出验证分数最高的k张图片

- 对查询图片进行粗特征提取
- 对查询图片进行关键点特征提取