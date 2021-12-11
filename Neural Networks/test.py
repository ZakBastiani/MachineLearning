import pandas as pd
import numpy as np
import nn

train_data_csv = "data/train.csv"
test_data_csv = "data/test.csv"

attributes = ['variance',
              'skewness',
              'curtosis',
              'entropy',
              'label']
train_df = pd.read_csv(train_data_csv, names=attributes).astype(float)
test_df = pd.read_csv(test_data_csv, names=attributes).astype(float)

train_df.loc[train_df['label'] == 0, 'label'] = -1
test_df.loc[test_df['label'] == 0, 'label'] = -1


# # Testing the NN with the example in the HW
# network = nn.NN(3, 2)
# network.weights_1 = np.array([[-1, 1], [-2, 2], [-3, 3]])
# network.weights_2 = np.array([[-1, 1], [-2, 2], [-3, 3]])
# network.weights_3 = np.array([[-1], [2], [-1.5]])
# network.forward(np.array([[1, 1]]))
# network.backward(1.0, 1.0)

widths = [5, 10, 25, 50, 100]

for i in widths:
    print('Width: ' + str(i))
    network = nn.NN(i, 4)
    network.train(train_df, 0.1, 10)
    pred = network.test(train_df.drop(columns=['label']))
    for i in range(len(pred)):
        if pred[i] > 0:
            pred[i] = 1
        else:
            pred[i] = -1
    error = sum(abs(pred - train_df['label'].to_numpy()))/(2*len(train_df))
    print('Train Error: ' + str(error))
    pred = network.test(test_df.drop(columns=['label']))
    for i in range(len(pred)):
        if pred[i] > 0:
            pred[i] = 1
        else:
            pred[i] = -1
    error = sum(abs(pred - test_df['label'].to_numpy()))/(2*len(test_df))
    print('Test Error: ' + str(error))






