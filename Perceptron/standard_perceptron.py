import numpy as np
import pandas as pd


class StandPerceptron:
    def __init__(self, train_data, T, lr):
        self.train_data = train_data
        self.train_data['bias'] = np.ones(len(train_data))
        self.w = np.zeros(len(self.train_data.iloc[0])-1)
        self.T = T
        self.lr = lr
        self.build()

    def build(self):
        for i in range(self.T):
            print(i)
            data = self.train_data.sample(len(self.train_data), replace=False, ignore_index=True)
            for j in range(len(data)):
                x = data.drop(columns=['label']).iloc[j].to_numpy()
                y = data['label'].iloc[j]
                if y * self.w.T@x <= 0:
                    self.w = self.w + self.lr * y * x

    def predict(self, test_data):
        pred = np.zeros(len(test_data))
        test_data['bias'] = np.ones(len(test_data))
        for i in range(len(test_data)):
            x = test_data.iloc[i].to_numpy()
            pred[i] = np.sign(self.w.T@x)
        test_data['pred'] = pred
        return test_data


