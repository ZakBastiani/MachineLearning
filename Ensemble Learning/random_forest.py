import numpy as np
import pandas as pd
import random_tree


class RandomForest:
    def __init__(self):
        self.train_data = []
        self.test_data = []

        self.first_test_data = []

    def run(self, train_data, test_data, attribute_types, T, sub_set_size):

        # Idk what this should be set to
        sample_size = len(train_data)

        train_data['f_pred'] = np.zeros(len(train_data))
        test_data['f_pred'] = np.zeros(len(train_data))
        train_data['pred'] = np.zeros(len(train_data))
        test_data['pred'] = np.zeros(len(train_data))
        train_data['Miss'] = np.zeros(len(train_data))
        test_data['Miss'] = np.zeros(len(train_data))

        train_err = np.zeros(T)
        test_err = np.zeros(T)

        for i in range(T):
            samp = train_data.sample(sample_size, replace=True, ignore_index=True)
            samp = samp.drop(columns=['pred', 'Miss', 'f_pred'])
            tree = random_tree.RandomDecisionTree(samp, attribute_types, 0, 100, sub_set_size)
            train_data = tree.testdata(train_data)
            test_data = tree.testdata(test_data)

            train_data['f_pred'] = train_data['f_pred'] + train_data['pred']
            test_data['f_pred'] = test_data['f_pred'] + test_data['pred']

            train_err[i] = sum(abs((train_data['Label'] - np.sign(train_data['f_pred']))/2))/len(train_data)
            test_err[i] = sum(abs((test_data['Label'] - np.sign(test_data['f_pred']))/2))/len(test_data)
            if i == 0:
                self.first_test_data = test_data.copy()

        self.train_data = train_data.copy()
        self.test_data = test_data.copy()
        return train_err, test_err
