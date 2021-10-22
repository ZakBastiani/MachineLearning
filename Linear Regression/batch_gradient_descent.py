import numpy as np
import pandas as pd


class BGD:
    def __init__(self, train_data, attributes, lr):
        self.w = np.zeros(len(attributes))
        self.learn_rate = []
        y = train_data.T.iloc[-1].to_numpy()
        train_data['bias'] = np.ones(len(train_data))
        train_data = train_data.drop(columns=['label'])
        x = train_data.to_numpy()
        diff = 1
        while diff > 0.000001:
            error = 0
            for i in range(len(x)):
                error += (y[i] - self.w.T@x[i])**2
            self.learn_rate.append(error)

            sum_w = np.zeros(len(attributes)).astype(float)
            for i in range(len(x)):
                sum_w += (y[i] - self.w.T@x[i])*(-x[i])
            self.w = self.w - lr * sum_w
            diff = np.linalg.norm(sum_w)



    def test(self, test_df):
        y = test_df.T.iloc[-1].to_numpy()
        test_df['bias'] = np.ones(len(test_df))
        test_df = test_df.drop(columns=['label'])
        x = test_df.to_numpy()
        error = 0
        for i in range(len(x)):
            error += (y[i] - self.w.T@x[i])**2
        return error
