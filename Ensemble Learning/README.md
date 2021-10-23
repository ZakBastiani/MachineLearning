This library implements adaboost, bagged trees and 
random trees. All files with the word test in them
are used to test the aforementioned algorithms. While run.sh
will run all of the test files the run time is extremely long.
To run any of the files simple type python3 followed by the name
of the file you want to run. If you are interested in running the files it is highly
suggested that you change the run parameters before hand. 
One of the most import parameters in terms of run time is T.
T is currently set to 500, but a T = 50 is suggested as at
point you basically made the best model possible.

In order to call any of the trees simply build the class and then use
the run command and inputting the train data frame, the test data frame,
the attribute list, and the T value. The T value determines the number 
of trees the algorithm will use. Note that with the random forest algorithm
you will also need to input the set size for the small random sets used to 
build each tree.