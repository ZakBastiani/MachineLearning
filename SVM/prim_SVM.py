import numpy as np


class PrimSvd:

    LR_TYPE_A = 0
    LR_TYPE_T = 1

    def __init__(self, train_data, T, lr, C, a, lr_type):
        self.train_data = train_data
        self.train_data['bias'] = np.ones(len(train_data))
        self.w = np.zeros(len(self.train_data.iloc[0])-1)
        self.T = T
        self.lr = lr
        if lr_type == self.LR_TYPE_A:
            self.lr_inc = self.lr_inc_a
        else:
            self.lr_inc = self.lr_inc_t
        self.C = C
        self.a = a
        self.build()

    def build(self):
        N = len(self.train_data)
        for i in range(self.T):
            lr_i = self.lr_inc(i)
            data = self.train_data.sample(len(self.train_data), replace=False, ignore_index=True)
            for j in range(len(data)):
                x = data.drop(columns=['label']).iloc[j].to_numpy()  # this could be done better
                y = data['label'].iloc[j]
                if y * self.w.T@x <= 1:
                    self.w = self.w - self.lr * np.append(self.w[:len(self.w)-1], 0) \
                             + self.lr * self.C * N * y * x
                else:
                    self.w[:len(self.w)-1] = (1-lr_i)*self.w[:len(self.w)-1]  # don't change the bias

    def lr_inc_t(self, t):
        return self.lr/(1+t)

    def lr_inc_a(self, t):
        return self.lr/(1 + (self.lr*t)/self.a)

    def predict(self, test_data):
        pred = np.zeros(len(test_data))
        test_data['bias'] = np.ones(len(test_data))
        for i in range(len(test_data)):
            x = test_data.iloc[i].to_numpy()
            pred[i] = np.sign(self.w.T@x)
        test_data['pred'] = pred
        return test_data


