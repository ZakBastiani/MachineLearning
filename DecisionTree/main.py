import numpy
import decisiontree
import pandas as pd

CSVfile = "Car_Data/train.csv"

with open(CSVfile, 'r') as f:
    for line in f:
        terms = line.strip().split(',')
        print(terms)

train_df = pd.read_csv(CSVfile)

print(train_df)