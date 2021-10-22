import numpy as np
import random_forest
import pandas as pd
import matplotlib.pyplot as plt

attribute_file = "data-desc.tex"
train_data_csv = "Bank_Data/train.csv"
test_data_csv = "Bank_Data/test.csv"

attributes = ['age',
              'job',
              'marital',
              'education',
              'default',
              'balance',
              'housing',
              'loan',
              'contact',
              'day',
              'month',
              'duration',
              'campaign',
              'pdays',
              'previous',
              'poutcome',
              'y']
attributes_types = {
    'age': ['numeric'],
    'job': ["admin.", "unknown", "unemployed", "management", "housemaid", "entrepreneur", "student",
            "blue-collar", "self-employed", "retired", "technician", "services"],
    'marital': ["married", "divorced", "single"],
    'education': ["unknown", "secondary", "primary", "tertiary"],
    'default': ['yes', 'no'],
    'balance': ['numeric'],
    'housing': ['yes', 'no'],
    'loan': ['yes', 'no'],
    'contact': ["unknown", "telephone", "cellular"],
    'day': ['numeric'],
    'month': ["jan", "feb", "mar", "apr", "may", "jun","jul", "aug", "sep", "oct", "nov", "dec"],
    'duration': ['numeric'],
    'campaign': ['numeric'],
    'pdays': ['numeric'],
    'previous': ['numeric'],
    'poutcome': ["unknown", "other", "failure", "success"],
    'y': ['yes', 'no']
}
train_df = pd.read_csv(train_data_csv, names=attributes)
test_df = pd.read_csv(test_data_csv, names=attributes)
for a in attributes:
    if attributes_types[a][0] == 'numeric':
        median = train_df[train_df[a] != 'unknown'][a].astype(float).median()
        train_df.loc[train_df[a].astype(float) > median, a] = median+1
        train_df.loc[train_df[a].astype(float) <= median, a] = median-1
        test_df.loc[test_df[a].astype(float) > median, a] = median+1
        test_df.loc[test_df[a].astype(float) <= median, a] = median-1
        attributes_types[a] = [median+1, median-1]

train_df = train_df.rename(columns={'y': 'Label'})
test_df = test_df.rename(columns={'y': 'Label'})
train_df['Label'] = train_df['Label'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)
test_df['Label'] = test_df['Label'].apply(lambda x: '1' if x == 'yes' else '-1').astype(float)

T = 500
forest = random_forest.RandomForest()
single_tree_bias = 0
single_tree_variance = 0

forest_bias = 0
forest_variance = 0

for i in range(100):
    samp = train_df.sample(1000, replace=False, ignore_index=True)
    out6 = forest.run(train_df, test_df, attributes_types, T, 6)
    forest_bias += sum((forest.test_data['Label'] - np.sign(forest.test_data['f_pred']))**2)/len(forest.test_data)
    single_tree_bias += sum((forest.first_test_data['Label'] - np.sign(forest.first_test_data['f_pred']))**2)/len(forest.first_test_data)

    forest_mean = forest.test_data['f_pred'].mean()
    st_mean = forest.first_test_data['f_pred'].mean()
    forest_variance += sum((forest_mean - np.sign(forest.test_data['f_pred']))**2)/(len(forest.test_data)-1)
    single_tree_variance += sum((forest_mean - np.sign(forest.first_test_data['f_pred']))**2)/(len(forest.first_test_data)-1)


N = len(test_df)
single_tree_bias = single_tree_bias/N
single_tree_variance = single_tree_variance/N
forest_bias = forest_bias/N
forest_variance = forest_variance/N

print('Single Tree Stats')
print('Bias: ' + str(single_tree_bias))
print('Variance: ' + str(single_tree_variance))
print('GSE: ' + str(single_tree_variance + single_tree_bias))
print()
print('Forest Stats')
print('Bias: ' + str(forest_bias))
print('Variance: ' + str(forest_variance))
print('GSE: ' + str(forest_variance + forest_bias))
