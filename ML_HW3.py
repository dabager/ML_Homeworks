import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

def printResults(testClassTable, predictions, method):
    print(method)
    print(confusion_matrix(testClassTable, predictions))  
    print(classification_report(testClassTable, predictions))  

def knn(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable):
    classifier = KNeighborsClassifier(n_neighbors=5)  
    classifier.fit(trainingFeatureTable, trainingClassTable)

    y_pred = classifier.predict(testFeatureTable)
    printResults(testClassTable, y_pred, 'KNN')


def decisionTree(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable):
    gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=200, min_samples_leaf=5) 
    gini.fit(trainingFeatureTable, trainingClassTable) 
    gini_pred = gini.predict(testFeatureTable)

    printResults(testClassTable, gini_pred, 'Decision Tree')

def svm(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable):
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(trainingFeatureTable, trainingClassTable) 
    y_pred = svclassifier.predict(testFeatureTable)
    printResults(testClassTable, y_pred, 'SVM')

def mlp(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(trainingFeatureTable, trainingClassTable)
    y_pred = clf.predict(testFeatureTable)
    printResults(testClassTable, y_pred, 'MLP')

def main():

    trainingData = "optdigits.tra"
    testData = "optdigits.tes"

    trainingTable = pd.read_csv(trainingData, names=['f01','f02','f03','f04','f05','f06','f07','f08','f09','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','class'])

    testTable = pd.read_csv(testData, names=['f01','f02','f03','f04','f05','f06','f07','f08','f09','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','class'])

    features = ['f01','f02','f03','f04','f05','f06','f07','f08','f09','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64']

    #x features y labels

    trainingFeatureTable = trainingTable.loc[:, features].values

    trainingClassTable = trainingTable.loc[:, ['class']].values

    testFeatureTable = testTable.loc[:, features].values

    testClassTable = testTable.loc[:, ['class']].values

    scaler = StandardScaler()
    scaler.fit(trainingFeatureTable)

    trainingFeatureTable = scaler.transform(trainingFeatureTable)
    testFeatureTable = scaler.transform(testFeatureTable)
    
    knn(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable)
    decisionTree(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable)
    svm(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable)
    mlp(trainingFeatureTable, trainingClassTable, testFeatureTable, testClassTable)

main()