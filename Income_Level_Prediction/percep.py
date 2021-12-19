import numpy as np
import pandas as pd
import standard_perceptron

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

for a in attributes:
    if attribute_types[a] == 'numeric':
        train_df = train_df.drop(columns=a)
        test_df = test_df.drop(columns=a)

# training_sets = np.array_split(train_df, 5)
#
# for i in range(5):
#     training_sets[i] = training_sets[i].drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'race', 'sex', 'native-country'])
#     squared = np.power(training_sets[i].drop(columns=['relationship', 'label']), 2)
#     squared = squared.rename(columns={'age': 'a2',
#                                       'fnlwgt': 'fnlw2',
#                                       'education-num': 'e2',
#                                       'capital-gain': 'cg2',
#                                       'capital-loss': 'cl2',
#                                       'hours-per-week': 'hpw2'})
#     training_sets[i] = pd.concat([training_sets[i], squared], axis=1)


# lr = 0.001
# acc = np.zeros((5, 5))
# for i in range(5):
#     stand = standard_perceptron.StandPerceptron(training_sets[i], 50, lr)
#     for l in range(5):
#         pred = stand.predict(training_sets[l].drop(columns=['label']))
#         error = sum(abs(pred - training_sets[l]['label'].to_numpy()))/(2*len(training_sets[l]))
#         acc[i][l] = error
#         print(error)
# print('Model:')
# print(acc)
# print(np.linalg.norm(acc))

# Changing categorical to numerical
train_df = pd.get_dummies(data=train_df, columns=['workclass', 'education', 'marital-status', 'occupation', 'race', 'relationship', 'native-country', 'sex'])
test_df = pd.get_dummies(data=test_df, columns=['workclass', 'education', 'marital-status', 'occupation', 'race', 'relationship', 'native-country', 'sex'])
test_df = pd.concat([test_df[train_df.drop(columns=['label']).columns], test_df['ID']], axis=1)

lr = 0.001
stand = standard_perceptron.StandPerceptron(train_df, 20, lr)
pred_train = stand.predict(train_df.drop(columns=['label']))
print(sum(abs(pred_train - train_df['label']))/(2*len(train_df)))

pred = stand.predict(test_df.drop(columns=['ID']))
for i in range(len(pred)):
    if pred[i] > 0:
        pred[i] = 1
    else:
        pred[i] = 0

predictions = pd.DataFrame({'ID': test_df['ID'], "Prediction": pred})

predictions.to_csv("prediction_per.csv", index=False)


