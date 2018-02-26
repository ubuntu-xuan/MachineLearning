# -*- coding:utf-8 -*-

from numpy import *
import operator
from os import listdir

def classify0(inX, dataSet, labels, k):
    dataSetSize = dataSet.shape[0]
    #tile(inX, (dataSetSize,1))：将inX向量按照(dataSetSize,1)重复生成新的矩阵
    #(dataSetSize,1)：dataSetSize：新矩阵的行  1：重复一次
    diffMat = tile(inX, (dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()  #argsort  数组值从小到大排列后再返回索引值
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]] #sortedDistIndicies[i]: 最小值的索引
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1 #利用字典，对应数字投票加1

    #按字典的item反向排序
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)

    #取出字典中第一个组对应的item
    return sortedClassCount[0][0]



def img2vector(filename):
    returnVect = zeros((1,1024))

    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        print 'i',i
        print 'lineStr',lineStr
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])

            print 'j',j
            print 'returnVect',returnVect
    print '--'*50
    return returnVect

def handwritingClassTest():
    hwLabels = []
    trainingFileList = listdir('./digits/trainingDigits')           #load the training set
    m = len(trainingFileList)
    trainingMat = zeros((m,1024))
    for i in range(m):
        fileNameStr = trainingFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        hwLabels.append(classNumStr)  #将训练集每个样本文件对应的类别记录下来

        #将所有的样本文件转换成向量再放在一个大矩阵里
        trainingMat[i,:] = img2vector('./digits/trainingDigits/%s' % fileNameStr)

    testFileList = listdir('./digits/testDigits')        #iterate through the test set
    errorCount = 0.0
    mTest = len(testFileList)
    for i in range(mTest):
        fileNameStr = testFileList[i]
        fileStr = fileNameStr.split('.')[0]     #take off .txt
        classNumStr = int(fileStr.split('_')[0])
        #将测试集的每个样本转换成向量
        vectorUnderTest = img2vector('./digits/testDigits/%s' % fileNameStr)
        classifierResult = classify0(vectorUnderTest, trainingMat, hwLabels, 3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, classNumStr)
        if (classifierResult != classNumStr): errorCount += 1.0
    print "\nthe total number of errors is: %d" % errorCount
    print "\nthe total error rate is: %f" % (errorCount/float(mTest))



def file2matrix(filename):
    with open(filename) as fr:
        arrayOLines = fr.readlines()
        numberOfLines = len(arrayOLines)         #get the number of lines in the file
        returnMat = zeros((numberOfLines,3))        #prepare matrix to return
        classLabelVector = []                       #prepare labels return
        index = 0
        for line in arrayOLines:
            line = line.strip()
            listFromLine = line.split('\t')
            returnMat[index,:] = listFromLine[0:3]
            classLabelVector.append(int(listFromLine[-1]))
            index += 1
        return returnMat,classLabelVector

def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m,1))
    normDataSet = normDataSet/tile(ranges, (m,1))   #element wise divide
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.10      #hold out 10%,10%作为测试集 ，90%为训练集
    datingDataMat,datingLabels = file2matrix('./datingTestSet2.txt')       #load data setfrom file
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0.0
    for i in range(numTestVecs):
        classifierResult = classify0(normMat[i,:],normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print "the classifier came back with: %d, the real answer is: %d" % (classifierResult, datingLabels[i])
        if (classifierResult != datingLabels[i]): errorCount += 1.0
    print "the total error rate is: %f" % (errorCount/float(numTestVecs))
    print errorCount



#datingClassTest()
handwritingClassTest()