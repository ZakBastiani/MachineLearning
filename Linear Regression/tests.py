import numpy as np
import pandas as pd
import batch_gradient_descent
import stochastic_gradient_decent
import analytical_gradient_descent
import matplotlib.pyplot as plt

train_data_csv = "concrete/train.csv"
test_data_csv = "concrete/test.csv"

attributes = ['Cement',
              'Slag',
              'Fly ash',
              'Water',
              'SP',
              'Coarse Aggr',
              'Fine Aggr',
              'label']
train_df = pd.read_csv(train_data_csv, names=attributes).astype(float)
test_df = pd.read_csv(test_data_csv, names=attributes).astype(float)

bgd = batch_gradient_descent.BGD(train_df, attributes, lr=0.01)
bgd_error = bgd.test(test_df)
print('BDG weights: ' + str(bgd.w))
print('BDG Error: ' + str(bgd_error))
sgd = stochastic_gradient_decent.SGD(train_df, attributes, lr=0.01)
sgd_error = sgd.test(test_df)
print('SDG weights: ' + str(sgd.w))
print('SDG Error: ' + str(sgd_error))


fig1 = plt.figure(1)
ax1 = plt.axes()
ax1.plot(range(len(bgd.learn_rate)), bgd.learn_rate, c='b', label='Cost')
ax1.set_title("Cost rate of Batch GD")

fig2 = plt.figure(2)
ax2 = plt.axes()
ax2.plot(range(len(sgd.learn_rate)), sgd.learn_rate, c='b', label='Cost')
ax2.set_title("Cost rate of Stochastic GD")

# Analytical solution
agd = analytical_gradient_descent.AGD(train_df, attributes)
agd_error = agd.test(test_df)
print('ADG weights: ' + str(agd.w))
print('ADG Error: ' + str(agd_error))

print('Error in BGD weight: ' + str(np.linalg.norm(bgd.w - agd.w)))
print('Error in SGD weight: ' + str(np.linalg.norm(sgd.w - agd.w)))

plt.show()
