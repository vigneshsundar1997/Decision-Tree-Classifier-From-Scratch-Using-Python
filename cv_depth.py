from typing import Deque
import pandas as pd
import numpy as np
from trees import decisionTree
from statistics import mean
from statistics import stdev
import timeit
import math
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel

#sample the data with the given random_state and t_frac
def sampleData(data,random_state_value,t_frac):
    return data.sample(frac=t_frac,random_state=random_state_value)

def performCV():
    trainData= pd.read_csv('trainingSet.csv')

    trainData = sampleData(trainData,18,1)

    halfTrainData = sampleData(trainData,32,0.5)
    #get the ten fold data
    tenFoldDataSet = np.array_split(halfTrainData,10)

    depth = [3,5,7,9]

    decisionTreeMeanAccuracies = []
    decisionTreeSE = []
    decisionTreeTenFoldAccuracyList = []


    for d in depth:
        print('Cross Validation running for depth', d)
        for index in range(10):
            temp_tenFoldDataSet=tenFoldDataSet.copy()

            setIndex = temp_tenFoldDataSet[index]
            del temp_tenFoldDataSet[index]

            setC = pd.concat(temp_tenFoldDataSet)

            training_accuracy,test_accuracy = decisionTree(setC,setIndex,d)
            decisionTreeTenFoldAccuracyList.append(test_accuracy)
        
        decisionTreeMeanAccuracies.append(mean(decisionTreeTenFoldAccuracyList))

        decisionTreeSE.append(stdev(decisionTreeTenFoldAccuracyList)/math.sqrt(10))
        
        decisionTreeTenFoldAccuracyList.clear()
        
    plt.errorbar( depth, decisionTreeMeanAccuracies, yerr= decisionTreeSE ,label='DT')
    
    plt.xlabel('Maximum Depth of the trees')
    plt.ylabel('Testing Accuracy')
    plt.legend()
    plt.title('Test Accuracy of DT,Bagging and RF and their standard errors')

    plt.show()

if __name__ == "__main__":
    performCV()