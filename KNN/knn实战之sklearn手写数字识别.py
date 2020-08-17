import numpy as np
import operator
from os import listdir
from sklearn.neighbors import KNeighborsClassifier as KNN

"""
函数说明:将32x32的二进制图像转换为1x1024向量。

Parameters:
    filename - 文件名
Returns:
    returnVect - 返回的二进制图像的1x1024向量

Modify:
    2020-08-17
"""
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0, 32*i+j] = int(lineStr[j])
    return returnVect


"""
函数说明:手写数字分类测试

Parameters:
    无
Returns:
    无

Modify:
    2020-8-17
"""
def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('trainingDigits')
    m = len(trainingFileList)
    trainingMat = np.zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        hwLabels.append(classNumber)
        trainingMat[i,:] = img2vector('trainingDigits/{}'.format(fileNameStr))
    neigh = KNN(n_neighbours=3, algorithm='auto')
    neigh.fit(trainingMat, hwLabels)
    testFileList = listdir('testDigits')
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        classNumber = int(fileNameStr.split('_')[0])
        vectorUnderTest = img2vector('testDigits/%s' % (fileNameStr))
        classifierResult = neigh.predict(vectorUnderTest)
        print("分类返回结果为%d\t真实结果为%d" % (classifierResult, classNumber))
        if classifierResult != classNumber:
            errorCount += 1.0
    print("总共错了%d个数据\n错误率为%f%%" % (errorCount, errorCount / mTest * 100))

"""
函数说明:main函数

Parameters:
    无
Returns:
    无

Modify:
    2020-8-17
"""
if __name__ == '__main__':
    handwritingClassTest()