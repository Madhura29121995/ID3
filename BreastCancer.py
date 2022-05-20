import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def CalculateEntropy(class1=0, class2=0):
    lisOfClass = [class1, class2]
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
        df = pd.DataFrame(dataset.groupby([header, columnOfClass])[columnOfClass].count())
        result = []
        for i in Counter(dataset[header]).keys():
            result.append(df.loc[i].values)

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
        index = {'diagnosis': 1, 'radius': 2, 'texture': 3, 'perimeter': 4, 'area': 5, 'smoothness': 6, 'compactness': 7, 'concavity': 8, 'concave points': 9}
        for i in t.keys():
            if i in index.keys():
                td = tupl[index[i]]
                s = t[i].get(tupl[index[i]], 0)
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
    occur = 0 # counter for cross validations performed
    AverageAcc = 0.0
    final_acc_arr = []
    std_dev = 0.0
    data = pd.read_csv('breastCancer.data', names=["radius", "texture", "perimeter", "area", "smoothness", "compactness", "concavity", "concave points", "symmetry", "fractal dimension", "Diagnosis"])

    #print (data.dtypes)
    ObjectColumns = data.select_dtypes(include=np.object).columns.tolist()
    data['concavity'] = pd.to_numeric(data['concavity'], errors='coerce')
    data = data.replace(np.nan, data['concavity'].mean(), regex=True)
    data["concavity"] = data["concavity"].astype(int)
    #print(data["concavity"].iloc[23])
    
    data['diagnosis'] = data['diagnosis'].astype(int)
    data['radius'] = data['radius'].astype(int)
    data['texture'] = data['texture'].astype(int)
    data['perimeter'] = data['perimeter'].astype(int)
    data['area'] = data['area'].astype(int)
    data['smoothness'] = data['smoothness'].astype(int)
    data['compactness'] = data['compactness'].astype(int)
    data['concavity'] = data['concavity'].astype(int)
    data['concave points'] = data['concave points'].astype(int)
    data['labels'] = data['labels'].astype(int)
    
    while (occur < 10):
        df = data.sample(frac=1)
        y = df["labels"]
        X = df.drop(["id", "labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.45)
        # CalculateEntropy of the entire training data set (y)
        setOfEntropy = CalculateEntropy(*[i for i in Counter(y_train).values()])
        print("The total CalculateEntropy of the training set is {}".format(setOfEntropy))
        model = ID3Algorithm() # creating a instance for the decision_tree class
        model.fitModule(X_train, y_train) # calling the fitModule method to create the tre
        accuracy_score(y_test, model.predict(X_test)) # the accuracy calculateScore under train-test-split
        acc_arr = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Accuracy Scores per ", occur+1, "Iteration is ", acc_arr)
        for i in range(0, len(acc_arr)):
            final_acc_arr.append(acc_arr[i])
        occur += 1
    AverageAcc = np.sum(final_acc_arr)/len(final_acc_arr)
    std_dev = np.std(final_acc_arr)
    print("Average Accuracy:", AverageAcc)
    print("Standard Deviation: ",std_dev)