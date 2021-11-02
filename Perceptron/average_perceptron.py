import numpy as np
import pandas as pd


class AvgPerceptron:
    def __init__(self, train_data, T, lr):
        self.train_data = train_data
        self.train_data['bias'] = np.ones(len(train_data))
        self.w = np.zeros(len(self.train_data.iloc[0])-1)
        self.a = self.w
        self.T = T
        self.lr = lr
        self.build()

    def build(self):
        for i in range(self.T):
            for j in range(len(self.train_data )):
                x = self.train_data.drop(columns=['label']).iloc[j].to_numpy()
                y = self.train_data['label'].iloc[j]
                if y * self.w.T@x <= 0:
                    self.w = self.w + self.lr * y * x
                self.a = self.a + self.w

    def predict(self, test_data):
        pred = np.zeros(len(test_data))
        test_data['bias'] = np.ones(len(test_data))
        for i in range(len(test_data)):
            x = test_data.iloc[i].to_numpy()
            pred[i] = np.sign(self.a.T@x)
        test_data['pred'] = pred
        return test_data
