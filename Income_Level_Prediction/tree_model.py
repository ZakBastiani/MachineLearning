import numpy as np
import pandas as pd
import decision_tree
import random_forest
import adaboost
import nn

attribute_file = "data-desc.tex"
train_data_csv = "Income_Data/train_final.csv"
test_data_csv = "Income_Data/test_final.csv"

train_df = pd.read_csv(train_data_csv)
test_df = pd.read_csv(test_data_csv)
attributes = ['age',
              'workclass',
              'fnlwgt',
              'education',
              'education-num',
              'marital-status',
              'occupation',
              'relationship',
              'race',
              'sex',
              'capital-gain',
              'capital-loss',
              'hours-per-week',
              'native-country']
attribute_types = {
    'age': ['numeric'],
    'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                  'Without-pay', 'Never-worked', '?'],
    'fnlwgt': ['numeric'],
    'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th',
                  '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool', '?'],
    'education-num': ['numeric'],
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                       'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                   'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                   'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
    'relationship': ['Wife', 'Husband', 'Not-in-family', 'Other-relative', 'Own-child', 'Unmarried'],
    'race': ['White', 'other'],
    'sex': ['Female', 'Male'],
    'capital-gain': ['numeric'],
    'capital-loss': ['numeric'],
    'hours-per-week': ['numeric'],
    'native-country': ['United-States', 'other']
}

train_df = train_df.rename(columns={'income>50K': 'label'})
train_df.loc[train_df['label'] == 0, 'label'] = -1

# Basically useless columns, eduction is already in num, and fnlwgt is uncorrelated with label
train_df = train_df.drop(columns=['education', 'fnlwgt'])
test_df = test_df.drop(columns=['education', 'fnlwgt'])

greater = train_df[train_df['label'] == 1]
lower = train_df[train_df['label'] == -1]
train_df = pd.concat([greater,  lower.sample(len(greater), replace=False, ignore_index=True)], axis=0)
train_df = train_df.sample(len(train_df), replace=False, ignore_index=True)

# Putting age into bins
ages = [0, 30, 60, 100]
labels = ['1', '2', '3']
train_df['age'] = pd.cut(train_df['age'], bins=ages, labels=labels)
test_df['age'] = pd.cut(test_df['age'], bins=ages, labels=labels)
attribute_types['age'] = labels

# Putting  hours worked into bins
hours = [0, 25, 40, 120]
labels = ['1', '2', '3']
train_df['hours-per-week'] = pd.cut(train_df['hours-per-week'], bins=hours, labels=labels)
test_df['hours-per-week'] = pd.cut(test_df['hours-per-week'], bins=hours, labels=labels)
attribute_types['hours-per-week'] = labels

# Bin education numbers
level = [0, 8, 14, 20]
labels = ['1', '2', '3']
train_df['education-num'] = pd.cut(train_df['education-num'], bins=level, labels=labels)
test_df['education-num'] = pd.cut(test_df['education-num'], bins=level, labels=labels)
attribute_types['education-num'] = labels

# Altering relationship
train_df.loc[train_df['relationship'] == 'Own-child', 'relationship'] = 'Unmarried'
train_df.loc[train_df['relationship'] == 'Other-relative', 'relationship'] = 'Unmarried'
train_df.loc[train_df['relationship'] == 'Not-in-family', 'relationship'] = 'Unmarried'
test_df.loc[test_df['relationship'] == 'Own-child', 'relationship'] = 'Unmarried'
test_df.loc[test_df['relationship'] == 'Other-relative', 'relationship'] = 'Unmarried'
test_df.loc[test_df['relationship'] == 'Not-in-family', 'relationship'] = 'Unmarried'
attribute_types['relationship'] = ['Wife', 'Husband', 'Unmarried']

# Alter occupation
train_df.loc[train_df['occupation'] == 'Prof-specialty', 'occupation'] = 'Exec-managerial'
train_df.loc[train_df['occupation'] != 'Exec-managerial', 'occupation'] = 'other'
test_df.loc[test_df['occupation'] == 'Prof-specialty', 'occupation'] = 'Exec-managerial'
test_df.loc[test_df['occupation'] != 'Exec-managerial', 'occupation'] = 'other'
attribute_types['occupation'] = ['Exec-managerial', 'other']

# Altering marital-status
train_df.loc[train_df['marital-status'] != 'Married-civ-spouse', 'marital-status'] = 'other'
test_df.loc[test_df['marital-status'] != 'Married-civ-spouse', 'marital-status'] = 'other'
attribute_types['marital-status'] = ['Married-civ-spouse', 'other']

# gains bins numbers
level = [-1, 100, 1000000]
labels = ['1', '2']
train_df['capital-gain'] = pd.cut(train_df['capital-gain'], bins=level, labels=labels)
test_df['capital-gain'] = pd.cut(test_df['capital-gain'], bins=level, labels=labels)
attribute_types['capital-gain'] = labels

# losses bins numbers
level = [-1, 200, 10000]
labels = ['1', '2']
train_df['capital-loss'] = pd.cut(train_df['capital-loss'], bins=level, labels=labels)
test_df['capital-loss'] = pd.cut(test_df['capital-loss'], bins=level, labels=labels)
attribute_types['capital-loss'] = labels

# Changing native country to either be USA or other
train_df.loc[train_df['native-country'] == 'Japan', 'native-country'] = 'United-States'
train_df.loc[train_df['native-country'] == 'Canada', 'native-country'] = 'United-States'
train_df.loc[train_df['native-country'] == 'India', 'native-country'] = 'United-States'
train_df.loc[train_df['native-country'] == 'Iran', 'native-country'] = 'United-States'
train_df.loc[train_df['native-country'] == 'Germany', 'native-country'] = 'United-States'
train_df.loc[train_df['native-country'] == 'England', 'native-country'] = 'United-States'
train_df.loc[train_df['native-country'] != 'United-States', 'native-country'] = 'other'
test_df.loc[test_df['native-country'] == 'Japan', 'native-country'] = 'United-States'
test_df.loc[test_df['native-country'] == 'Canada', 'native-country'] = 'United-States'
test_df.loc[test_df['native-country'] == 'India', 'native-country'] = 'United-States'
test_df.loc[test_df['native-country'] == 'Iran', 'native-country'] = 'United-States'
test_df.loc[test_df['native-country'] == 'Germany', 'native-country'] = 'United-States'
test_df.loc[test_df['native-country'] == 'England', 'native-country'] = 'United-States'
test_df.loc[test_df['native-country'] != 'United-States', 'native-country'] = 'other'

# Changing race to be white or other
train_df.loc[train_df['race'] != 'White', 'race'] = 'other'
test_df.loc[test_df['race'] != 'White', 'race'] = 'other'

# Lets split the data into 5 parts
training_sets = np.array_split(train_df, 5)
depth = 7
entropy = 0

# acc = np.zeros((5, 5))
# for i in range(5):
#     tree = decision_tree.DecisionTree(training_sets[i], attribute_types, entropy, depth)
#     for l in range(5):
#         pred = tree.testdata(training_sets[l].drop(columns=['label']))
#         error = sum(abs(pred - training_sets[l]['label'].to_numpy()))/(2*len(training_sets[l]))
#         print('Train Error ' + str(l) + ': ' + str(error))
#         acc[i][l] = error
# print(acc)
# print(np.linalg.norm(acc))

# tree = decision_tree.DecisionTree(train_df, attribute_types, entropy, depth)
# pred_train = tree.testdata(train_df.drop(columns=['label']))
rand_forest = random_forest.RandomForest()
pred_train, pred = rand_forest.run(train_df, test_df.drop(columns=['ID']), attribute_types, 50, 6)
print(sum(abs(pred_train - train_df['label']))/(2*len(train_df)))
#
# pred = tree.testdata(test_df.drop(columns=['ID']))
for i in range(len(pred)):
    if pred[i] > 0:
        pred[i] = 1
    else:
        pred[i] = 0

predictions = pd.DataFrame({'ID': test_df['ID'], "Prediction": pred})

predictions.to_csv("prediction_tree.csv", index=False)
