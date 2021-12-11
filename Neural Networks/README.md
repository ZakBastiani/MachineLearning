This files builds one Neural Network. The neural network will only have three layers but can have any number of nodes in the hidden layers.

NN.__init__ takes in the number of nodes in the hidden layer, and dim which is the number of columns in the data

NN.forward takes in a data sample and returns its prediction

NN.backwards takes in the real y for the previous forward and the current learning rate and updates the NN

NN.train trains the NN on a dataset, and uses the lr_0 and d which controls the rate at which the NN updates

NN.test returns predictions for the input dataset