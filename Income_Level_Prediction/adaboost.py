import numpy as np
import pandas as pd
import decision_tree


class Adaboost:
    def run(self, train_data, test_data, attribute_types, T):

        # Idk what this should be set to
        sample_size = len(train_data)

        d_t = np.ones(len(train_data))
        d_t = d_t/len(train_data)
        train_data['prob'] = d_t
        train_data['f_pred'] = np.zeros(len(train_data))
        test_data['f_pred'] = np.zeros(len(test_data))
        train_data['pred'] = np.zeros(len(train_data))
        test_data['pred'] = np.zeros(len(test_data))
        train_data['Miss'] = np.zeros(len(train_data))
        test_data['Miss'] = np.zeros(len(test_data))

        for i in range(T):
            samp = train_data.sample(sample_size, replace=True, weights=train_data['prob'], ignore_index=True)
            samp = samp.drop(columns=['prob', 'Miss', 'pred', 'f_pred'])
            tree = decision_tree.DecisionTree(samp, attribute_types, 0, 1)
            train_data = tree.testdata(train_data)
            test_data = tree.testdata(test_data)

            w_error = sum(train_data['prob']*train_data['Miss'])
            alpha = 0.5 * np.log((1-w_error)/(w_error+0.000000001))

            train_data['f_pred'] = train_data['f_pred'] + alpha * train_data['pred']
            test_data['f_pred'] = test_data['f_pred'] + alpha * test_data['pred']

            train_data['prob'] = train_data['prob'] * np.exp(-alpha * train_data['income>50K']*train_data['pred'])
            train_data['prob'] = train_data['prob']/sum(train_data['prob'])

        return test_data




