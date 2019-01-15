import math
import pandas as pd
from IPython.display import display, HTML
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
import matplotlib.pyplot as plt
import numpy as np
import logging

trainingData = "optdigits.tra"
testData = "optdigits.tes"

table = pd.read_csv(trainingData, names=['f01','f02','f03','f04','f05','f06','f07','f08','f09','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64','class'])

features = ['f01','f02','f03','f04','f05','f06','f07','f08','f09','f10','f11','f12','f13','f14','f15','f16','f17','f18','f19','f20','f21','f22','f23','f24','f25','f26','f27','f28','f29','f30','f31','f32','f33','f34','f35','f36','f37','f38','f39','f40','f41','f42','f43','f44','f45','f46','f47','f48','f49','f50','f51','f52','f53','f54','f55','f56','f57','f58','f59','f60','f61','f62','f63','f64']

featureTable = table.loc[:, features].values

classTable = table.loc[:, ['class']].values

featureTable = StandardScaler().fit_transform(featureTable)

lda = LDA(n_components=2)

featureComponents = lda.fit_transform(featureTable, classTable)

principalTable = pd.DataFrame(data = featureComponents, columns = ['F1', 'F2'])

finalTable = pd.concat([principalTable, table[['class']]], axis = 1)

rowCount = finalTable.shape[0]

errorCount = 0

for i in range(rowCount):
    pc1i = finalTable.iloc[i]['F1']
    pc2i = finalTable.iloc[i]['F2']
    classi = finalTable.iloc[i]['class']
    
    finalTable['dif1'] = finalTable['F1'] - pc1i
    finalTable['dif2'] = finalTable['F2'] - pc2i
    finalTable['sqrt'] = np.sqrt(pow(finalTable['dif1'],2) + pow(finalTable['dif2'],2))
    tempTable = finalTable.drop(finalTable.index[i])
    minIndex = tempTable['sqrt'].idxmin()
    ##logging.debug(i)
    if(classi != finalTable.iloc[minIndex]['class']):
        errorCount += 1

errorPercentage = round(((errorCount / rowCount) * 100), 2)
print('Error : ' + str(errorPercentage) + ' %')

fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Feature 1', fontsize = 15)
ax.set_ylabel('Feature 2', fontsize = 15)
ax.set_title('2 component LDA', fontsize = 20)
targets = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
colors = ['#4285f4', '#ea4335', '#fbbc05', '#34a853', '#673ab7', '#b882f1', '#e29df4', '#fa8585', '#fbadd4', '#f75cab']
for target, color in zip(targets,colors):
    indicesToKeep = finalTable['class'] == target
    ax.scatter(finalTable.loc[indicesToKeep, 'F1']
               , finalTable.loc[indicesToKeep, 'F2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()

plt.show()
