import numpy as np
from scipy.optimize import minimize


class DualSVM:

    LINEAR = 0
    GAUSSIAN = 1

    def __init__(self, train_data, C, kern_type, theta):
        self.x = train_data.drop(columns=['label']).to_numpy()
        self.y = np.expand_dims(train_data['label'].to_numpy(), axis=0).T
        self.C = C
        self.w = np.zeros(len(self.x))
        self.a = np.zeros(len(train_data))
        self.b = 0
        self.theta = theta
        if kern_type == self.LINEAR:
            self.kern = self.dot
        else:
            self.kern = self.gaussian
        self.build()

    def build(self):
        cons = [{'type': 'eq', 'fun': lambda a: a.T@self.y},
                {'type': 'ineq', 'fun': lambda a: self.C*np.ones(len(a)) - a}]

        def fun(a):
            out = -(sum(a) - 0.5*sum(sum(np.expand_dims(self.a, axis=0)@np.expand_dims(self.a, axis=0).T
                                         *self.y@self.y.T*self.kern(self.x, self.x))))
            return out

        res = minimize(fun,
                       self.a,
                       method='SLSQP',
                       constraints=cons)
        self.a = res.x
        self.w = sum(np.expand_dims(self.a, axis=0).T * self.y * self.x)
        self.b = sum(np.expand_dims(self.a, axis=0).T * self.y)


    @staticmethod
    def dot(x1, x2):
        return x1@x2.T

    def gaussian(self, x1, x2):
        return np.exp((- (np.tile(x1.T[0], (len(x1), 1)) - np.tile(x2.T[0], (len(x1), 1)).T) ** 2
                       - (np.tile(x1.T[1], (len(x1), 1)) - np.tile(x2.T[1], (len(x1), 1)).T) ** 2
                       - (np.tile(x1.T[2], (len(x1), 1)) - np.tile(x2.T[2], (len(x1), 1)).T) ** 2
                       - (np.tile(x1.T[3], (len(x1), 1)) - np.tile(x2.T[3], (len(x1), 1)).T) ** 2)/self.theta)

    def predict(self, test_data):
        pred = np.zeros(len(test_data))
        for i in range(len(test_data)):
            x_i = test_data.iloc[i].to_numpy()
            # pred[i] = np.sign(self.w.T@x_i + self.b)
            kern_x = self.kern(self.x, x_i)
            holder = sum(sum(np.expand_dims(self.a, axis=0)*self.y.T*kern_x)) + self.b
            pred[i] = np.sign(holder)
        test_data['pred'] = pred
        return test_data


