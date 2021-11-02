All of the Perceptron algorithms take three arguments, first the training data in a pandas
dataframe with is label column as 'label' using -1 and 1 for true and false. Second is the T value
for how many cycles each algorithm should do. Third is the learning rate for the algorithm.
Lastly the algorithms use the predict method to predict labels for an input test data dataframe.
The test data dataframe should not have a 'label' column. The predict method returns the test dataframe
with an added 'pred' column with each row's predicted label.

run.sh is set up to run the test.py file. This file builds and predicts the Bank_data with all three of
the algorithms. Weights and prediction accuracy and then printed to standard out.