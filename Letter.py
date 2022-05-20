import sys
import pandas as pd
from collections import Counter
from math import log
import numpy as np
from sklearn.base import BaseEstimator as estimator, ClassifierMixin as mix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

def CalculateEntropy(class1=0, class2=0, class3=0, class4=0, class5=0, class6=0, class7=0, class8=0, class9=0, class10=0,
            class11=0, class12=0, class13=0, class14=0, class15=0, class16=0, class17=0, class18=0, class19=0, class20=0,
            class21=0, class22=0, class23=0, class24=0, class25=0, class26=0):
    lisOfClass = [class1, class2, class3, class4, class5, class6, class7, class8, class9, class10,
            class11, class12, class13, class14, class15, class16, class17, class18, class19, class20,
            class21, class22, class23, class24, class25, class26]
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
        index = {'x-box': 1, 'y-box': 2, 'width': 3, 'high': 4, 'onpix': 5, 'x-bar': 6, 'y-bar': 7, 'x2bar': 8, 'y2bar': 9, 'xybar': 10, 'x2ybr' :11, 'xy2br': 12, 'x-ege': 13, 'xegvy': 14, 'y-ege': 15, 'yegvx': 16}
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
    data = pd.read_csv('letter-recognition.data', names=["lettr", "x-box", "y-box", "width", "high", "onpix", "x-bar", "y-bar", "x2bar", "y2bar", "xybar", "x2ybr", "xy2br", "x-ege", "xegvy", "y-ege", "yegvx"])

    print (data.dtypes)
    ObjectColumns = data.select_dtypes(include=np.object).columns.tolist()
    data['lettr'] = [ord(item)-64 for item in data['lettr']]
    print(data["lettr"].iloc[23])

    while (occur < 10):
        df = data.sample(frac=1)
        y = df["labels"]
        X = df.drop(["labels"], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
        
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