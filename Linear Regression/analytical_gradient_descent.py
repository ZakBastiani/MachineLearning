import numpy as np
import pandas as pd


class AGD:
    def __init__(self, train_data, attributes):
        self.w = np.zeros(len(attributes))
        y = train_data.T.iloc[-1].to_numpy()
        train_data['bias'] = np.ones(len(train_data))
        train_data = train_data.drop(columns=['label'])
        x = train_data.to_numpy()
        self.w = np.linalg.inv(x.T@x)@x.T@y.T

    def test(self, test_df):
        y = test_df.T.iloc[-1].to_numpy()
        test_df['bias'] = np.ones(len(test_df))
        test_df = test_df.drop(columns=['label'])
        x = test_df.to_numpy()
        error = 0
        for i in range(len(x)):
            error += (y[i] - self.w.T@x[i])**2
        return error
