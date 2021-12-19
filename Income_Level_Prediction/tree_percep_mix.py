import numpy as np
import pandas as pd
import decision_tree
import dualSVM

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

svm_train = train_df.drop(columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'svm_pred'])
tree_train = train_df.drop(columns=['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])


nn_test = test_df.drop(columns=['ID', 'workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country', 'svm_pred'])
tree_test = test_df.drop(columns=['ID', 'age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week'])

svm = dualSVM.DualSVM(train_df, C, 1, theta)
svm_pred = svm.predict(svm_train.drop(columns=['label']))

error = 0
for j in range(len(svm_pred)):
    if svm_pred[j] > 0:
        error += abs(1 - train_df.iloc['label', j])
    else:
        error += abs(0 - train_df.iloc['label', j])
error = error/(len(train_df))
print('Train Error: ' + str(error))

tree_train['svm_pred'] = np.sign(svm_pred)
tree = decision_tree.DecisionTree(tree_train, attribute_types, 0, 5)
print(tree.root.attribute)

nn_pred = svm.predict(nn_test)
tree_test['svm_pred'] = np.sign(nn_pred)
pred = tree.testdata(tree_test)['pred'].to_numpy()

for j in range(len(pred)):
    if pred[j] > 0:
        pred[j] = 1
    else:
        pred[j] = 0

predictions = pd.DataFrame({'ID': test_df['ID'], "Prediction": pred})

predictions.to_csv("prediction.csv", index=False)


