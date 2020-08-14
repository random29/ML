# -*- coding: utf-8 -*-
"""
Created on Fri May  5 09:58:16 2017

@author: Сабина
"""

import numpy as np
import sklearn
from sklearn import datasets
from matplotlib import pyplot as plt
from sklearn import tree
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dig = datasets.load_digits()

bestAcc = [0] * 100
idex = np.arange(1000)
np.random.shuffle(idex)
xTr = dig.data[idex[:600]]
xVal = dig.data[idex[600:800]]
xTest = dig.data[idex[800:]]

yTr = dig.target[idex[:600]]
yVal = dig.target[idex[600:800]]
yTest = dig.target[idex[800:]]

#max_depth = np.random.rand(10, 100, 10)
#model = DecisionTreeClassifier()
#model.fit(xTrain, yTrain)

#maxDepth = np.random.rand(100)

for i in range (100):
       
    maxDepth = np.arange(1, 101)
    model = DecisionTreeClassifier(criterion = "gini",
                               max_depth=maxDepth[i], splitter='best')
    model.fit(xTr, yTr)
    bestAcc[i]=sklearn.metrics.accuracy_score(yVal, model.predict(xVal))
    
#print (bestAcc) 
#print (max(bestAcc))

#clf_entropy = DecisionTreeClassifier(criterion = "entropy", 
 #                                    max_depth=10, splitter='best')
#clf_entropy = DecisionTreeClassifier(criterion = "entropy", 
 #                                    max_depth=50, splitter='best')
clf_entropy = DecisionTreeClassifier(criterion = "entropy", 
                                     max_depth=100, splitter='best')
clf_entropy.fit(xTr, yTr)

yPredictVald = model.predict(xVal)
yPredictEntropyVald = clf_entropy.predict(xVal)


yPredictTest = model.predict(xTest)
yPredictEntropyTest = clf_entropy.predict(xTest)

print ("DTaccuracyVald = " ,sklearn.metrics.accuracy_score(yVal, yPredictVald))
print ("DTaccuracyEntropyVald = " ,sklearn.metrics.accuracy_score(yVal, yPredictEntropyVald))
print ("DTaccuracyTest = " , sklearn.metrics.accuracy_score(yTest, yPredictTest))
print ("DTaccuracyEntropyTest = " , sklearn.metrics.accuracy_score(yTest, yPredictEntropyTest))


#RandomForest

modelForest = RandomForestClassifier(criterion = "gini",
                               max_depth=100)
modelForest.fit(xTr, yTr)

#modelForest = RandomForestClassifier(criterion = "gini",
#                               max_depth=10)
#modelForest = RandomForestClassifier(criterion = "gini",
#                               max_depth=50)

clf_entropyForest = RandomForestClassifier(criterion = "entropy", 
                                     max_depth=100)
clf_entropyForest.fit(xTr, yTr)

yPredictValdForest = modelForest.predict(xVal)
yPredictEntropyValdForest = clf_entropyForest.predict(xVal)

yPredictTestForest = modelForest.predict(xTest)
yPredictEntropyTestForest = clf_entropyForest.predict(xTest)

print ("RFaccuracyTest = " , sklearn.metrics.accuracy_score(yTest, yPredictTestForest))
print ("RFaccuracyEntropyTest = " , sklearn.metrics.accuracy_score(yTest, yPredictEntropyTestForest))






















