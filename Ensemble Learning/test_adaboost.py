import numpy as np
import adaboost
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
ada = adaboost.Adaboost()
out = ada.run(train_df, test_df, attributes_types, T)

x = range(1, T+1)
fig1 = plt.figure(1)
ax1 = plt.axes()
ax1.plot(x, out[0], c='b', label='Train Accuracy')
ax1.plot(x, out[1], c='r', label='Test Accuracy')
ax1.set_title("Tree Error")
plt.legend()

fig2 = plt.figure(2)
ax2 = plt.axes()
ax2.plot(x, out[2], c='b', label='Train Accuracy')
ax2.plot(x, out[3], c='r', label='Test Accuracy')
ax2.set_title("Adaboost Error")
plt.legend()
plt.show()

print(out)
