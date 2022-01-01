import pandas as pd
import numpy as np
from trees import bagging, decisionTree, randomForests
from statistics import mean,stdev
import math
import matplotlib.pyplot as plt
import timeit
from scipy.stats import ttest_rel


#sample the data based on the given random_state and t_frac
def sampleData(data,random_state_value,t_frac):
    return data.sample(frac=t_frac,random_state=random_state_value)

def performCV():
    trainData= pd.read_csv('trainingSet.csv')
    trainData = sampleData(trainData,18,1)
    #get the ten fold data
    tenFoldDataSet = np.array_split(trainData,10)

    t_fracs = [0.05, 0.075, 0.1, 0.15, 0.2]

    decisionTreeMeanAccuracies = []
    decisionTreeSE = []
    decisionTreeTenFoldAccuracyList = []

    for t_frac in t_fracs:
        print("Cross Validation running for t_frac", t_frac)
        for index in range(10):
  
            temp_tenFoldDataSet=tenFoldDataSet.copy()

            setIndex = temp_tenFoldDataSet[index]
            del temp_tenFoldDataSet[index]

            setC = pd.concat(temp_tenFoldDataSet)

            train_set = sampleData(setC,32,t_frac)

            training_accuracy,test_accuracy = decisionTree(train_set,setIndex,8)
            decisionTreeTenFoldAccuracyList.append(test_accuracy)

            training_accuracy,test_accuracy = bagging(train_set,setIndex,8,30)
            
            training_accuracy,test_accuracy = randomForests(train_set,setIndex,8,30)

        decisionTreeMeanAccuracies.append(mean(decisionTreeTenFoldAccuracyList))
        
        decisionTreeSE.append(stdev(decisionTreeTenFoldAccuracyList)/math.sqrt(10))
        
        
        decisionTreeTenFoldAccuracyList.clear()
        
    #get the training set size for each of the fraction to be plotted in x-axis of the graph
    plot_x_axis = [t_frac * setC.shape[0] for t_frac in t_fracs]

    plt.errorbar( plot_x_axis, decisionTreeMeanAccuracies, yerr= decisionTreeSE ,label='DT')
    
    plt.xlabel('Training Dataset Size')
    plt.ylabel('Testing Accuracy')
    plt.legend()
    plt.title('Test Accuracy of DT,BG and RF and their standard errors')

    plt.show()

if __name__ == "__main__":
    performCV()