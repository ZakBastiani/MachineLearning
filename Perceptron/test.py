import numpy as np
import pandas as pd
import standard_perceptron
import voted_perceptron
import average_perceptron

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

lr = 0.00001

stand = standard_perceptron.StandPerceptron(train_df, 10, lr)
pred = stand.predict(test_df.drop(columns=['label']))

print('Standard Acc: ' + str(sum(abs(pred['pred'] - test_df['label']))/2/len(test_df)))

voted = voted_perceptron.VotedPerceptron(train_df, 10, lr)
pred = voted.predict(test_df.drop(columns=['label']))

print('Voted Acc: ' + str(sum(abs(pred['pred'] - test_df['label']))/2/len(test_df)))

avg = average_perceptron.AvgPerceptron(train_df, 10, lr)
pred = avg.predict(test_df.drop(columns=['label']))

print('Average Acc: ' + str(sum(abs(pred['pred'] - test_df['label']))/2/len(test_df)))

