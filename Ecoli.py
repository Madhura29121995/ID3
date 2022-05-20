import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def CalculateEntropy(class1=0, class2=0, class3=0, class4=0, class5=0, class6=0, class7=0, class8=0):
    lisOfClass = [class1, class2, class3, class4, class5, class6, class7, class8]
    findFinalEntropy = 0
    for c in lisOfClass:
        if c != 0:
            findFinalEntropy += -((c / sum(lisOfClass)) * log(c / sum(lisOfClass), 4))
    return findFinalEntropy

class ID3Algorithm(estimator, mix):

    def __init__(self, columnOfClass="labels"):
        self.columnOfClass = columnOfClass

    @staticmethod
    def calculateScore(noOfSlipt, entro, total):
        setOfEntropy = [CalculateEntropy(*i) for i in noOfSlipt] 
        f = lambda x, y: (sum(x) / total) * y
        result = [f(i, j) for i, j in zip(noOfSlipt, setOfEntropy)]
        return entro - sum(result)

    @staticmethod
    def splittingSet(header, dataset, columnOfClass):
        data = pd.DataFrame(dataset.groupby([header, columnOfClass])[columnOfClass].count())
        result = []
        for i in Counter(dataset[header]).keys():
            result.append(data.loc[i].values)

        return result

    @classmethod
    def findingNode(cls, dataset, columnOfClass):
        entro = CalculateEntropy(*[i for i in Counter(dataset[columnOfClass]).values()])
        result = {}  
        for i in dataset.columns:
            if i != columnOfClass:
                noOfSlipt = cls.splittingSet(i, dataset, columnOfClass)
                gainScore = cls.calculateScore(noOfSlipt, entro, total=len(dataset)) 
                result[i] = gainScore
        return max(result, key=result.__getitem__)

    @classmethod
    def recursionMethod(cls, dataset, tree, columnOfClass):
        n = cls.findingNode(dataset, columnOfClass)  
        branches = [i for i in Counter(dataset[n])]
        tree[n] = {}
        for j in branches:  
            br_data = dataset[dataset[n] == j]  
            if CalculateEntropy(*[i for i in Counter(br_data[columnOfClass]).values()]) != 0:
                tree[n][j] = {}
                cls.recursionMethod(br_data, tree[n][j], columnOfClass)
            else:
                r = Counter(br_data[columnOfClass])
                tree[n][j] = max(r, key=r.__getitem__)  
        return

    @classmethod
    def prediction(cls, tupl, t):
       
        if type(t) is not dict:
            return t
        index = {'mcg': 1, 'gvh': 2, 'lip': 3, 'chg': 4, 'aac': 5, 'alm1': 6, 'alm2': 7}
        for i in t.keys():
            if i in index.keys():
                td = tupl[index[i]]
                s = t[i].get(tupl[index[i]], 0)
                r = cls.prediction(tupl, t[i].get(tupl[index[i]], 0))
        return r

    # main prediction function
    def predict(self, test):
        result = []
        for i in test.itertuples():
            result.append(ID3Algorithm.prediction(i, self.tree_))
        return pd.Series(result) 

    def fitModule(self, X, y):  
        columnOfClass = self.columnOfClass 
        dataset = X.assign(labels=y)
        self.tree_ = {} 
        ID3Algorithm.recursionMethod(dataset, self.tree_, columnOfClass)
        return self


if __name__ == '__main__':
    occur = 0  
    AverageAcc = 0.0
    final_acc_arr = []
    std_dev = 0.0
    data = pd.read_csv('ecoli.data', names=["Sequence", "mcg", "gvh", "lip", "chg", "aac", "alm1", "alm2", "decision"])

    data["decision"].replace(["cp","im","imU","imS","imL","om","omL","pp"], [0,1,2,3,4,5,6,7], inplace = True)
    data['mcg'] = [int(item) for item in data['mcg']]
    data['gvh'] = [int(item) for item in data['gvh']]
    data['lip'] = [int(item) for item in data['lip']]
    data['chg'] = [int(item) for item in data['chg']]
    data['aac'] = [int(item) for item in data['aac']]
    data['alm1'] = [int(item) for item in data['alm1']]
    data['alm2'] = [int(item) for item in data['alm2']]
    
    while (occur < 10):
        data = data.sample(frac=1)
        y = data["labels"]
        X = data.drop(["labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        
        setOfEntropy = CalculateEntropy(*[i for i in Counter(y_train).values()])
        print("The total CalculateEntropy of the training set is {}".format(setOfEntropy))
        model = ID3Algorithm()  
        model.fitModule(X_train, y_train)  
        accuracy_score(y_test, model.predict(X_test))  
        acc_arr = cross_val_score(model, X, y, cv=2, scoring='accuracy')
        print("Accuracy Scores per ", occur + 1, "Iteration is ", acc_arr)
        for i in range(0, len(acc_arr)):
            final_acc_arr.append(acc_arr[i])
        occur += 1
    AverageAcc = np.sum(final_acc_arr) / len(final_acc_arr)
    std_dev = np.std(final_acc_arr)
    print("Average Accuracy:", AverageAcc)
    print("Standard Deviation: ", std_dev)