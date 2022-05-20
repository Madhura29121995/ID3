import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def CalculateEntropy(class1=0, class2=0, class3=0, class4=0):
    lisOfClass = [class1, class2, class3, class4]
    findFinalEntropy = 0
    for c in lisOfClass:
        if c != 0:
            findFinalEntropy += -((c/sum(lisOfClass))*log(c/sum(lisOfClass), 4))
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
        if type(t) is int:
            return "NaN" 
        elif type(t) is not dict:
            return t
        index = {'buying': 1, 'maint': 2, 'doors': 3, 'persons': 4, 'lug_boot': 5, 'safety': 6}
        for i in t.keys():
            if i in index.keys():
                s = t[i].get(tupl[index[i]])
                r = cls.prediction(tupl, t[i].get(tupl[index[i]], 0))
        return r

   
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
    data = pd.read_csv('car.data', names=["buying", "maint", "doors", "persons", "lug_boot", "safety", "decision"])

    data["decision"].replace(["unacc", "acc", "good", "vgood"], [0, 1, 2, 3], inplace = True)
    data["safety"].replace(["low", "med", "high"], [0, 1, 2], inplace = True)
    data["lug_boot"].replace(["small", "med", "big"], [0, 1, 2], inplace = True)
    data["persons"].replace(["more"], [4], inplace = True)
    data["doors"].replace(["5more"], [6], inplace = True)
    data["maint"].replace(["vhigh", "high", "med", "low"], [4, 3, 2, 1], inplace = True)
    data["buying"].replace(["vhigh", "high", "med", "low"], [4, 3, 2, 1], inplace = True)

    data['decision'] = data['decision'].astype(int)
    data['safety'] = data['safety'].astype(int)
    data['lug_boot'] = data['lug_boot'].astype(int)
    data['persons'] = data['persons'].astype(int)
    data['doors'] = data['doors'].astype(int)
    data['maint'] = data['maint'].astype(int)
    data['buying'] = data['buying'].astype(int)
    while (occur < 10):
        data = data.sample(frac=1)
        y = data["labels"]
        X = data.drop(["labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)
        
        setOfEntropy = CalculateEntropy(*[i for i in Counter(y_train).values()])
        print("The total CalculateEntropy of the training set is {}".format(setOfEntropy))
        model = ID3Algorithm() 
        model.fitModule(X_train, y_train)
        accuracy_score(y_test, model.predict(X_test))
        acc_arr = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Accuracy Scores per ", occur+1, "Iteration is ", acc_arr)
        for i in range(0, len(acc_arr)):
            final_acc_arr.append(acc_arr[i])
        occur += 1
    AverageAcc = np.sum(final_acc_arr)/len(final_acc_arr)
    std_dev = np.std(final_acc_arr)
    print("Average Accuracy:", AverageAcc)
    print("Standard Deviation: ",std_dev)