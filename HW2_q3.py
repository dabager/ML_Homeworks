import math
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import logging
import copy

trainingData = "optdigits.tra"
testData = "optdigits.tes"

table = pd.read_csv(trainingData, names=['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','class'])
rowCount = table.shape[0]

features = ['f1','f2','f3','f4','f5','f6','f7','f8','f9','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64']

featureTable = table.loc[:, features].values

classTable = table.loc[:, ['class']].values

clusterCount = 20

clusterCenters = [[0 for x in range(64)] for y in range(clusterCount)]

clusterList = []
clusterListOld = []

errorCount = 0;
errorList = []

for i in range(clusterCount):
##    index = np.random.randint(0, rowCount - 1)
    clusterList.append((i, []))
    clusterListOld.append((i, []))
    for j in range(1, 65):
        clusterCenters[i][j - 1] = float(table.iloc[i]['f' + str(j)])
        
iteration = 0;
maxNumberOfIterations = 50;
lastErrorCounter = 15


def clusterComparer():
    if (iteration is 0):
        return True
    value = False
    for i in range(clusterCount):
        if (clusterList[i][1] != clusterListOld[i][1]):
            value = True
            break
    return value

def lastErrorChecker():
    if (iteration < 5):
        return True
    else:
        if ((float(sum(errorList[-lastErrorCounter:])) / lastErrorCounter) == errorPercentage):
            return False
        else:
            return True

while (clusterComparer() and lastErrorChecker() and iteration < maxNumberOfIterations):
    errorCount = 0
    print('iteration ' + str(iteration))
    for i in range(clusterCount):
        table['powsum' + str(i)] = 0
        for j in range(1, 65):
            table['powsum' + str(i)] = table['powsum' + str(i)] + pow(table['f' + str(j)] - clusterCenters[i][j - 1], 2)
        table['sqrt' + str(i)] = np.sqrt(table['powsum' + str(i)])
        table = table.drop('powsum' + str(i), axis = 1)

    clusterTable = table.drop(features, axis = 1)
    clusterTable = clusterTable.drop('class', axis = 1)

    clusterTable['cluster'] = clusterTable.idxmin(axis=1)
    clusterTable['cluster'] = clusterTable['cluster'].str.replace('sqrt','')

    for i in range(clusterCount):
        clusterTable = clusterTable.drop('sqrt' + str(i), axis = 1)
        table = table.drop('sqrt' + str(i), axis = 1)

    mergedTable = pd.concat([table, clusterTable[['cluster']]], axis = 1)

    for i in range(clusterCount):
        clusterListOld[i][1].clear()
        
        for cluster in clusterList:
            clusterList[i][1].append(cluster)
            
        clusterList[i][1].clear()
        
        clfilter = mergedTable[mergedTable['cluster'] == str(i)]
        
        for j in range(1, 65):
            avg = clfilter['f' + str(j)].mean()
            clusterCenters[i][j - 1] = avg

    for i in range(rowCount):
        cluster = int(mergedTable.iloc[i]['cluster'])
        clusterList[cluster][1].append(i)

    for i in range(clusterCount):        
        clfilter = mergedTable[mergedTable['cluster'] == str(i)]
        values = clfilter.groupby(['class']).size().sort_values(ascending=False)

        maxLabelCount = values.iloc[0]
        totalCount = clfilter.shape[0]

        errorCount += (totalCount - maxLabelCount)

    errorPercentage = round(((errorCount / rowCount) * 100), 4)
    errorList.append(errorPercentage)

            
    iteration += 1

print('Final Error : ' + str(errorPercentage) + ' %')

