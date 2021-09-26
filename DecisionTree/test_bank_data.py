import numpy as np
import decisiontree
import pandas as pd

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

# changing unknowns to the most common attribute in the training set
for a in attributes:
    if "unknown" in attributes_types[a]:
        avg = train_df.loc[train_df[a] != "unknown"][a].mode()[0]
        train_df.loc[train_df[a] == "unknown", a] = avg
        test_df.loc[test_df[a] == "unknown", a] = avg
        attributes_types[a].remove("unknown")


acc = np.zeros((6, 16))

for i in range(1, 17):
    print(i)
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.IG_ID, i)
    acc[0, i-1] = tree.testdata(train_df)
    acc[1, i-1] = tree.testdata(test_df)
print("done with ig")

for i in range(1, 17):
    print(i)
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.ME_ID, i)
    acc[2, i-1] = tree.testdata(train_df)
    acc[3, i-1] = tree.testdata(test_df)
print("done with me")

for i in range(1, 17):
    print(i)
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.GI_ID, i)
    acc[4, i-1] = tree.testdata(train_df)
    acc[5, i-1] = tree.testdata(test_df)
print("done with gi")

for i in range(16):
    print(str(i + 1) + " & "
          + str("{:.4f}".format(acc[0, i])) + " & "
          + str("{:.4f}".format(acc[1, i])) + " & "
          + str("{:.4f}".format(acc[2, i])) + " & "
          + str("{:.4f}".format(acc[3, i])) + " & "
          + str("{:.4f}".format(acc[4, i])) + " & "
          + str("{:.4f}".format(acc[5, i])) + " \\\\ \hline")
