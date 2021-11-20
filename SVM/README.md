This folder contains two SVM methods. Both of them follow a stand set up of declearing the model and passing in the test
data, followed by asking to make predictions.

primSVM: 

init{
train_data: data to train the model on  

T: number of cyclces

lr: learning rate

C: your C variable

a: a variable effecting the learning rate

lr_type: 0 for when A is used, 1 for when only t is used}

pred{input_data: the test data set, this should not include the label for the data}

dualSVM

init{train_data: data to train the model on  

C: your C variable

kern_type: 0 for linear kernel, 1 for gaussian kernel

theta: bandwidth on the gaussian kernel}

pred{input_data: the test data set, this should not include the label for the data}