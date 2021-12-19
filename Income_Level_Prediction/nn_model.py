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


# Changing categorical to numerical
train_df = pd.get_dummies(data=train_df, columns=['workclass', 'education', 'marital-status', 'occupation', 'race', 'relationship', 'native-country', 'sex'])
test_df = pd.get_dummies(data=test_df, columns=['workclass', 'education', 'marital-status', 'occupation', 'race', 'relationship', 'native-country', 'sex'])
test_df = pd.concat([test_df[train_df.drop(columns=['label']).columns], test_df['ID']], axis=1)


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


network = nn.NN(25, 107)
network.train(train_df, 0.0001, 10)
nn_pred = np.sign(network.test(train_df.drop(columns=['label'])))
print(sum(abs(nn_pred - train_df['label']))/(2*len(train_df)))

pred = network.test(test_df.drop(columns=['ID']))
for j in range(len(pred)):
    if pred[j] > 0:
        pred[j] = 1
    else:
        pred[j] = 0

predictions = pd.DataFrame({'ID': test_df['ID'], "Prediction": pred})

predictions.to_csv("prediction_nn.csv", index=False)
