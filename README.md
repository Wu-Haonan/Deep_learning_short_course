<p align="center">
    <img src="https://github.com/Wu-Haonan/Deep_learning_short_course/tree/main/IMG" width="100%">
</p>

这个代码仓库为博客的深度学习短课提供练习服务，[深度学习短课博客链接](https://wu-haonan.github.io/2022/01/10/Why_and_what.html)。

# MLP练习

MLP练习在文件夹./Drug_Toxicity/下面，./Drug_Toxicity/Data_file/文件夹中存放了药物（训练集文件train_feature.pkl，测试集文件test_feature.pkl）和相应的hERG(心脏安全性评价的二分类标签)（文件hERG.pkl）[^1]

具体而言，train_feature.pkl文件存放了1974个药物样本，每个样本由729个特征构成，hERG.pkl文件存放了每个药物样本的二分类标签。我们需要用这两个数据进行训练。然后，test_feature.pkl文件存放了测试集的样本，我们需要用生成好的模型来预测其标签。

读者可以运行hERG_train.py文件和hERG_test.py文件对数据进行训练和预测，相应的结果以及模型保存在./Drug_Toxicity/hERG/文件夹中

[^1]:[数据来源于华为杯2021数学建模D题](https://cpipc.acge.org.cn//cw/detail/4/2c9080147c73b890017c7779e57e07d2)
