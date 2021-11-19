import pandas as pd
import numpy as np
import prim_SVM

train_data_csv = "Bank_Data/train.csv"
test_data_csv = "Bank_Data/test.csv"

attributes = ['variance',
              'skewness',
              'curtosis',
              'entropy',
              'label']
train_df = pd.read_csv(train_data_csv, names=attributes).astype(float)
test_df = pd.read_csv(test_data_csv, names=attributes).astype(float)

train_df.loc[train_df['label'] == 0, 'label'] = -1
test_df.loc[test_df['label'] == 0, 'label'] = -1

lr = 0.001
a = 0.001
T = 100
C_list = [100/873, 500/873, 700/873]
weights = []

# print('Using learning rate change with a:')
# for C in C_list:
#     prim = prim_SVM.PrimSvd(train_df, T, lr, C, a, 0)
#     pred_train = prim.predict(train_df.drop(columns=['label']))
#     pred_test = prim.predict(test_df.drop(columns=['label']))
#
#     print('C: ' + str(C))
#     print('Train Error: ' + str(sum(abs(pred_train['pred'] - train_df['label'])) / (2 * len(train_df))))
#     print('Test Error: ' + str(sum(abs(pred_test['pred'] - test_df['label'])) / (2 * len(test_df))))
#     weights.append(prim.w)
#     print()
#
# print('Using learning rate change with t:')
# for C in C_list:
#     prim = prim_SVM.PrimSvd(train_df, T, lr, C, a, 1)
#     pred_train = prim.predict(train_df.drop(columns=['label']))
#     pred_test = prim.predict(test_df.drop(columns=['label']))
#
#     print('C: ' + str(C))
#     print('Train Error: ' + str(sum(abs(pred_train['pred'] - train_df['label'])) / (2 * len(train_df))))
#     print('Test Error: ' + str(sum(abs(pred_test['pred'] - test_df['label'])) / (2 * len(test_df))))
#     weights.append(prim.w)
#     print()
#
# print('Weight Differences')
# for i in range(len(C_list)):
#     print(str(C_list[i]) + ': ' + str(np.linalg.norm(weights[i] - weights[i+3])))
