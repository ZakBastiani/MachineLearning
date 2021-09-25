import numpy
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
all_data = pd.concat([train_df, test_df])
print("IG Gain:")
for i in range(1, 7):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.IG_ID, i)
    acc = tree.testdata(all_data)
    print("Depth : " + str(i) + ", Accuracy: " + str(acc))

print("ME Gain:")
for i in range(1, 7):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.ME_ID, i)
    acc = tree.testdata(all_data)
    print("Depth : " + str(i) + ", Accuracy: " + str(acc))

print("GI Gain:")
for i in range(1, 7):
    tree = decisiontree.DecisionTree(train_df, attributes_types, decisiontree.DecisionTree.GI_ID, i)
    acc = tree.testdata(all_data)
    print("Depth : " + str(i) + ", Accuracy: " + str(acc))

