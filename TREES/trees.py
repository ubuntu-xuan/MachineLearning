# -*- coding:utf-8 -*-
__author__ = 'xuan'
from math import log
import operator


def createDataSet():
    dataSet = [[1,1,'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

'''求熵'''
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys(): labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        #求和
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

'''
    划分数据集：按照最优特征划分数据集
    @dataSet: 待划分的数据集
    @axis: 划分数据集的特征
    @value： 特征的取值
'''

def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            print('axis',axis)
            print('values',value)
            # print('featVec',featVec)
            reducedFeatVec = featVec[:axis]
            # print('reducedFeatVec',reducedFeatVec)
            #切片  [0:1] 0到1个元素 不包含第1个
            #比如a=[1,2,3],b=[4,5,6],则a.extend(b)=[1,2,3,4,5,6]
            reducedFeatVec.extend(featVec[axis+1:])
            #比如a=[1,2,3],b=[4,5,6],则a.extend(b)=[1,2,3,[4,5,6]]
            retDataSet.append(reducedFeatVec)
    print('retDataSet',retDataSet)
    return retDataSet

'''
    @return int bestFeature 最好特征对应列的索引
'''

def chooseBestFeatureToSplit(dataSet):
    #特征数量 = 数据第一行减去最后一列
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain = 0.0
    bestFeature = -1
    for i in range(numFeatures):
        # print('对第%i个特征',i+1)
        #分别取出每列特征
        featList = [example[i] for example in dataSet]
        #放入集合 去重
        uniqueVals = set(featList)
        newEntropy = 0.0
        for value in uniqueVals:
            # print('分集',value)
            subDataSet = splitDataSet(dataSet,i,value)
            #每个分集的样本数 / 样本总数
            prob = len(subDataSet)/float(len(dataSet))
            #划分数据集后的熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        #信息增益
        infoGain = baseEntropy - newEntropy
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = i
    # print('bset',bestFeature)
    return bestFeature

'''
    如果数据集已经处理了所有属性，但类标签依然
    不是唯一的，此时我们需要如何定义该叶子节点，
    通常采用多数表决的方法
'''
def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    sortedClassCount = sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]


'''
     创建树
'''

def createTree(dataSet,labels):
    classList = [example[-1] for example in dataSet]
    #若数据集所有类别相同则停止所有划分
    if classList.count(classList[0]) ==  len(classList):
        print('all labels same')
        return classList[0]
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    bestFeat = chooseBestFeatureToSplit(dataSet)
    print('bestFeat',bestFeat)
    bestFeatLabel = labels[bestFeat]
    print('bestFeatLabel',bestFeatLabel)

    myTree = {bestFeatLabel:{}}
    del(labels[bestFeat])
    print('leave labels',labels)
    #取出最好特征的对应那列
    featValues = [example[bestFeat] for example in dataSet]
    print('featValues',featValues)
    uniqueVals = set(featValues)
    for value in uniqueVals:
        print('class__',value)
        subLabels = labels[:]
        print('subLabels',subLabels)
        myTree[bestFeatLabel][value] = createTree(splitDataSet(dataSet,bestFeat,value),subLabels)
        print('myTree',myTree)
    return myTree


if __name__ == '__main__':
    myDat,labels = createDataSet()
    # chooseBestFeatureToSplit(myDat)
    createTree(myDat,labels)