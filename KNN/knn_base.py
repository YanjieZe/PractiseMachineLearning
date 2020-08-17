"""
    功能：实现了简单的KNN算法
    时间：2020.8.4
    作者：来自CSDN
    抄写者：yanjie Ze
"""


import numpy as np
import matplotlib.pyplot as plt
import operator

#创建数据集
def createDataset():
    #四组二维特征
    group = np.array([[1,101],[5,89],[108,5],[115,8]])
    #四组特征的标签
    labels = ['爱情片','爱情片','动作片','动作片']
    return group, labels


def classify0(inX, dataSet, labels, k):
    # 得到dataset的行数
    dataSetSize = dataSet.shape[0]

    #用tile函数对矩阵进行横向和纵向的复制,并减去原dataset
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet

    #平方
    sqDiffMat = diffMat**2

    # sum()所有元素相加，sum(0)列相加，sum(1)行相加
    sqDistances = sqDiffMat.sum(axis=1)

    #开方，求得距离
    distances = sqDistances**0.5

    #从小到大排序.参数已默认,返回索引
    sortedDistIndices = distances.argsort()

    #定一个记录类别次数的字典
    classCount = {}
    for i in range(k):
        #取出前k个元素的类别
        votelabel = labels[sortedDistIndices[i]]

        # dict.get(key,default=None),字典的get()方法,返回指定键的值,如果值不在字典中返回默认值。
        # 计算类别次数
        classCount[votelabel] = classCount.get(votelabel, 0)+1

    # python3中用items()替换python2中的iteritems()
    # key=operator.itemgetter(1)根据字典的值进行排序
    # key=operator.itemgetter(0)根据字典的键进行排序
    # reverse降序排序字典
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)

    # 返回次数最多的类别,即所要分类的类别
    return sortedClassCount[0][0]




if __name__ == '__main__' :
    #自己创建数据集
    group, labels = createDataset()
    #测试集
    test = [101,20]
    #KNN分类
    test_class = classify0(test,group, labels, 3)
    print(test_class)
