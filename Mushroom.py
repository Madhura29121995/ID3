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
        index = {'cap-shape': 1, 'cap-surface': 2, 'cap-color': 3, 'bruises': 4, 'odor': 5, 'gill-attachment': 6,
                 'gill-spacing': 7, 'gill-size': 8, 'gill-color': 9, 'stalk-shape': 10, 'stalk-root': 11, 'stalk-surface-above-ring': 12, 'stalk-surface-below-ring': 13,
    'stalk-color-above-ring': 14, 'stalk-color-below-ring': 15, 'veil-type': 16, 'veil-color': 17, 'ring-number': 18, 'ring-type': 19, 'spore-print-color': 20,
    'population': 21, 'habitat': 22}
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
    occur = 0  
    AverageAcc = 0.0
    final_acc_arr = []
    std_dev = 0.0
    header_row = ["labels", "cap-shape", "cap-surface", "cap-color", "bruises", "odor",  "gill-attachment",
    "gill-spacing", "gill-size", "gill-color", "stalk-shape", "stalk-root", "stalk-surface-above-ring", "stalk-surface-below-ring",
    "stalk-color-above-ring", "stalk-color-below-ring", "veil-type", "veil-color", "ring-number", "ring-type", "spore-print-color",
    "population", "habitat"]  
    mushroom_df = pd.read_csv(r'C:\Users\Soorya\Desktop\CS6735-MachineLearning\Prog Project\mushroom.data',
                            delimiter=",", names=header_row) 

    mushroom_df.replace(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z'],[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25],inplace = True)

    mushroom_df["stalk-root"].replace(["?"], ['0'], inplace = True)
    mushroom_df["stalk-root"] = mushroom_df['stalk-root'].astype(int)
    average = mushroom_df["stalk-root"].mean()
    mushroom_df["stalk-root"].replace([0], [average], inplace = True)

    mushroom_df["labels"] = mushroom_df['labels'].astype(int)
    mushroom_df["cap-shape"] = mushroom_df['cap-shape'].astype(int)
    mushroom_df["cap-surface"] = mushroom_df['cap-surface'].astype(int)
    mushroom_df["cap-color"] = mushroom_df['cap-color'].astype(int)
    mushroom_df["bruises"] = mushroom_df['bruises'].astype(int)
    mushroom_df["odor"] = mushroom_df['odor'].astype(int)
    mushroom_df["gill-attachment"] = mushroom_df['gill-attachment'].astype(int)
    mushroom_df["gill-spacing"] = mushroom_df['gill-spacing'].astype(int)
    mushroom_df["gill-size"] = mushroom_df['gill-size'].astype(int)
    mushroom_df["gill-color"] = mushroom_df['gill-color'].astype(int)
    mushroom_df["stalk-shape"] = mushroom_df['stalk-shape'].astype(int)
    mushroom_df["stalk-surface-above-ring"] = mushroom_df['stalk-surface-above-ring'].astype(int)
    mushroom_df["stalk-surface-below-ring"] = mushroom_df['stalk-surface-below-ring'].astype(int)
    mushroom_df["stalk-color-above-ring"] = mushroom_df['stalk-color-above-ring'].astype(int)
    mushroom_df["stalk-color-below-ring"] = mushroom_df['stalk-color-below-ring'].astype(int)
    mushroom_df["veil-type"] = mushroom_df['veil-type'].astype(int)
    mushroom_df["veil-color"] = mushroom_df['veil-color'].astype(int)
    mushroom_df["ring-number"] = mushroom_df['ring-number'].astype(int)
    mushroom_df["ring-type"] = mushroom_df['ring-type'].astype(int)
    mushroom_df["spore-print-color"] = mushroom_df['spore-print-color'].astype(int)
    mushroom_df["population"] = mushroom_df['population'].astype(int)
    mushroom_df["habitat"] = mushroom_df['habitat'].astype(int)

    while (occur < 10):
        df = mushroom_df.sample(frac=1)
        y = df["labels"]
        X = df.drop(["labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
        
        setOfEntropy = CalculateEntropy(*[i for i in Counter(y_train).values()])
        print("The total CalculateEntropy of the training set is {}".format(setOfEntropy))
        model = ID3Algorithm() 
        model.fitModule(X_train, y_train) 
        accuracy_score(y_test, model.predict(X_test)) 

        acc_arr = cross_val_score(model, X, y, cv=5, scoring='accuracy')
        print("Accuracy Scores per ", occur + 1, "Iteration is ", acc_arr)
        for i in range(0, len(acc_arr)):
            final_acc_arr.append(acc_arr[i])
        occur += 1
    AverageAcc = np.sum(final_acc_arr) / len(final_acc_arr)
    std_dev = np.std(final_acc_arr)
    print("Average Accuracy:", AverageAcc)
    print("Standard Deviation: ", std_dev)