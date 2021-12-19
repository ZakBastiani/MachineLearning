import numpy as np
import pandas as pd
import decision_tree
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
    'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
    'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty',
                   'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                   'Priv-house-serv', 'Protective-serv', 'Armed-Forces', '?'],
    'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
    'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
    'sex': ['Female', 'Male'],
    'capital-gain': ['numeric'],
    'capital-loss': ['numeric'],
    'hours-per-week': ['numeric'],
    'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece', 'South', 'China', 'Cuba', 'Iran',
                       'Honduras', 'Philippines', 'Italy', 'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland', 'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
                       'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland', 'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong', 'Holand-Netherlands', '?'],
    'nn_pred': ['numeric']
}

train_df = train_df.rename(columns={'income>50K': 'label'})
train_df.loc[train_df['label'] == 0, 'label'] = -1

# # Changing numerical to categorical
# for a in attributes:
#     if attribute_types[a][0] == 'numeric':
#         median = train_df[train_df[a] != 'unknown'][a].astype(float).median()
#         train_df.loc[train_df[a].astype(float) > median, a] = median+1
#         train_df.loc[train_df[a].astype(float) <= median, a] = median-1
#         test_df.loc[test_df[a].astype(float) > median, a] = median+1
#         test_df.loc[test_df[a].astype(float) <= median, a] = median-1
#         attribute_types[a] = [median+1, median-1]
#
# # Testing without some columns
# train_df = train_df.drop(columns=['native-country'])
# test_df = test_df.drop(columns=['native-country'])

# # Changing categorical to numerical
# for a in attributes:
#     if attribute_types[a][0] != 'numeric':
#         counter = 0.0
#         for att in attribute_types[a]:
#             train_df.loc[train_df[a] == att, a] = counter
#             counter += 1.0
#         train_df[a] = train_df[a].astype(float)
#
# train_df = train_df.sample(len(train_df), replace=False, ignore_index=True)
#
# # Testing without some columns
# train_df = train_df.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])
# test_df = test_df.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])


# # Lets split the data into 5 parts
# training_sets = np.array_split(train_df, 5)
#
# acc = np.zeros((5, 5))
# for i in range(5):
#     network = nn.NN(20, 6)
#     network.train(training_sets[i], 0.00001, 5)
#     for l in range(5):
#         pred = network.test(training_sets[l].drop(columns=['label']))
#         for j in range(len(pred)):
#             if pred[j] > 0:
#                 pred[j] = 1
#             else:
#                 pred[j] = -1
#         error = sum(abs(pred - training_sets[l]['label'].to_numpy()))/(2*len(training_sets[l]))
#         print('Train Error ' + str(l) + ': ' + str(error))
#         acc[i][l] = error

# # Lets split the data into 5 parts
train_df['nn_pred'] = np.zeros(len(train_df))
train_df = train_df.reindex(columns=['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
                          'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week',
                          'native-country', 'nn_pred', 'label'])
test_df['nn_pred'] = np.zeros(len(test_df))

# training_sets = np.array_split(train_df, 5)
# nn_training_sets = []
# tree_training_sets = []
#
# for i in range(5):
#     nn_training_sets.append(training_sets[i].drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'nn_pred']))
#     tree_training_sets.append(training_sets[i].drop(columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']))
#
# acc = np.zeros((5, 5))
# for i in range(5):
#     network = nn.NN(20, 6)
#     network.train(nn_training_sets[i], 0.00001, 5)
#     nn_pred = network.test(nn_training_sets[i].drop(columns=['label']))
#     tree_training_sets[i]['nn_pred'] = np.sign(nn_pred)
#     tree = decision_tree.DecisionTree(tree_training_sets[i], attribute_types, 0, 5)
#     for l in range(5):
#         nn_pred = network.test(nn_training_sets[l].drop(columns=['label']))
#         tree_training_sets[l]['nn_pred'] = np.sign(nn_pred)
#         pred = tree.testdata(tree_training_sets[l].drop(columns=['label']))['pred'].to_numpy()
#         for j in range(len(pred)):
#             if pred[j] > 0:
#                 pred[j] = 1
#             else:
#                 pred[j] = -1
#         error = sum(abs(pred - training_sets[l]['label'].to_numpy()))/(2*len(training_sets[l]))
#         acc[i][l] = error
#         print(error)
#
# print('Model:')
# print(acc)
# print(np.linalg.norm(acc))

nn_train = train_df.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'nn_pred'])
tree_train = train_df.drop(columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])


nn_test = test_df.drop(columns=['ID', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'nn_pred'])
tree_test = test_df.drop(columns=['ID', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

network = nn.NN(10, 6)
network.train(nn_train, 0.0001, 10)
nn_pred = network.test(nn_train.drop(columns=['label']))
tree_train['nn_pred'] = np.sign(nn_pred)
tree = decision_tree.DecisionTree(tree_train, attribute_types, 0, 5)
print(tree.root.attribute)

nn_pred = network.test(nn_test)
tree_test['nn_pred'] = np.sign(nn_pred)
pred = tree.testdata(tree_test)['pred'].to_numpy()

for j in range(len(pred)):
    if pred[j] > 0:
        pred[j] = 1
    else:
        pred[j] = 0

predictions = pd.DataFrame({'ID': test_df['ID'], "Prediction": pred})

predictions.to_csv("prediction.csv", index=False)


