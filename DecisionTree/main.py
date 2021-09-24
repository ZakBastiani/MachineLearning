import numpy
import decisiontree
import pandas as pd

attribute_file = "data-desc.tex"
train_data_csv = "Car_Data/train.csv"
test_data_csv = "Car_Data/test.csv"

attributes = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']
train_df = pd.read_csv(train_data_csv, names=attributes)
test_df = pd.read_csv(test_data_csv, names=attributes)


