import numpy
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

all_data = pd.concat([train_df, test_df])

tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.IG_ID, 16)
acc = tree.testdata(all_data)
print(acc)

print("IG Gain:")
for i in range(1, 16):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.IG_ID, i)
    acc = tree.testdata(all_data)
    print("Depth : " + str(i) + ", Accuracy: " + str(acc))

print("ME Gain:")
for i in range(1, 16):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.ME_ID, i)
    acc = tree.testdata(all_data)
    print("Depth : " + str(i) + ", Accuracy: " + str(acc))

print("GI Gain:")
for i in range(1, 16):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.GI_ID, i)
    acc = tree.testdata(all_data)
    print("Depth : " + str(i) + ", Accuracy: " + str(acc))