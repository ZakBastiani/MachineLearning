import numpy as np
import pandas as pd
from DecisionTree import decisiontree

class Adaboost:
    def __init__(self, train_data, test_data, attribute_types, T):

        # Idk what this should be set to
        sample_size = len(train_data)

        d_t = np.ones(len(train_data))
        d_t = d_t/len(train_data)
        train_data['prob'] = d_t

        for i in range(T):
            samp = train_data.sample(sample_size, replace=True, weights=train_data['prob'])
            tree = decisiontree.DecisionTree(samp, attribute_types, 0, 1)
            p_data = tree.testdata(train_data)
            p_test_data = tree.testdata(test_data)
            w_error = sum(p_data['prob']*p_data['Miss'])
            alpha = 0.5 * np.log((1-w_error)/w_error)
            train_data['prob'] = train_data['prob'] * np.exp(-alpha*p_data['Label']*p_data['pred'])
            train_data['prob'] = train_data['prob']/sum(train_data['prob'])



