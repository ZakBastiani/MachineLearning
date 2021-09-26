import numpy as np
import decisiontree
import pandas as pd

attribute_file = "data-desc.tex"
train_data_csv = "Car_Data/train.csv"
test_data_csv = "Car_Data/test.csv"

attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
attributes_types = {
    'buying': ["vhigh", "high", "med", "low"],
    'maint': ["vhigh", "high", "med", "low"],
    'doors': ["2", "3", "4", "5more"],
    'persons': ["2", "4", "more"],
    'lug_boot': ["small", "med", "big"],
    'safety': ["low", "med", "high"],
    'label': ["unacc", "acc", "good", "vgood"]
}
train_df = pd.read_csv(train_data_csv, names=attributes)
test_df = pd.read_csv(test_data_csv, names=attributes)
acc = np.zeros((6, 6))

for i in range(1, 7):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.IG_ID, i)
    acc[0, i-1] = tree.testdata(train_df)
    acc[1, i-1] = tree.testdata(test_df)

for i in range(1, 7):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.ME_ID, i)
    acc[2, i-1] = tree.testdata(train_df)
    acc[3, i-1] = tree.testdata(test_df)

for i in range(1, 7):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.GI_ID, i)
    acc[4, i-1] = tree.testdata(train_df)
    acc[5, i-1] = tree.testdata(test_df)

for i in range(6):
    print(str(i + 1) + " & "
          + str("{:.4f}".format(acc[0, i])) + " & "
          + str("{:.4f}".format(acc[1, i])) + " & "
          + str("{:.4f}".format(acc[2, i])) + " & "
          + str("{:.4f}".format(acc[3, i])) + " & "
          + str("{:.4f}".format(acc[4, i])) + " & "
          + str("{:.4f}".format(acc[5, i])) + " \\\\ \hline")


