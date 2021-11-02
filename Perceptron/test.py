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

lr = 0.000001

stand = standard_perceptron.StandPerceptron(train_df, 10, lr)
pred = stand.predict(test_df.drop(columns=['label']))

print('Standard Acc: ' + str(sum(abs(pred['pred'] - test_df['label']))/2/len(test_df)))
print(stand.w)

voted = voted_perceptron.VotedPerceptron(train_df, 10, lr)
pred = voted.predict(test_df.drop(columns=['label']))

print('Voted Acc: ' + str(sum(abs(pred['pred'] - test_df['label']))/2/len(test_df)))

np.savetxt('w_list.csv', voted.w_list, delimiter=',')
np.savetxt('c_list.csv', voted.c_list, delimiter=',')
form = "{:.4f}"
for i in range(len(voted.w_list)):
    output = form.format(voted.c_list[i]) + ' & '
    for j in range(len(voted.w_list[i])-1):
        output += form.format(voted.w_list[i][j]) + ' & '
    output += form.format(voted.w_list[i][len(voted.w_list[i])-1]) + '\\\\ \\hline'
    print(output)


avg = average_perceptron.AvgPerceptron(train_df, 10, lr)
pred = avg.predict(test_df.drop(columns=['label']))

print('Average Acc: ' + str(sum(abs(pred['pred'] - test_df['label']))/2/len(test_df)))
print(avg.a)


