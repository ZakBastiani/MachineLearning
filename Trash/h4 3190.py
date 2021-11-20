import pandas as pd
import numpy as np

# a = pd.read_csv('data/A.csv', header=None).to_numpy()
# centering = np.zeros(len(a))
# for i in range(len(a)):
#     centering[i] = sum(a[i])/len(a)
# cent = np.ones((35, 8))
# for i in range(len(centering)):
#     cent[i] = centering[i]*cent[i]
# a = a - cent
# u, s, v = np.linalg.svd(a, full_matrices=False)
#
# s2 = s**2
# for i in range(len(s)):
#     out = "{:.2f}".format(s2[i]) + ': ['
#     for j in range(len(u.T)):
#         out += "{:.4f}".format(u.T[i][j])
#         if j != len(u.T)-1:
#             out += ', '
#     print(out + '] \\\\')

x_data = pd.read_csv('data/X4.csv', names=[0, 1, 2]).to_numpy()
y_data = pd.read_csv('data/Y4.csv', names=[0]).to_numpy()

w = np.array([0, 0, 0])
lr = 0.02
#
for j in range(50):
    for i in range(len(x_data)):
        grad = 2*(-x_data[i])*(y_data[i] - w.T@x_data[i])
        w = w - lr*grad
        loss = 0
    for l in range(len(x_data)):
        loss += (y_data[l] - w.T@x_data[l])**2
    print('Loss: ' + str(loss) +
          ', Grad Norm: ' + str(np.linalg.norm(grad)) +
          ', New Weight: ' + str(w) + ' \\\\')


loss = 0
for i in range(1):
    loss += (y_data[i] - w.T@x_data[i])**2

print(loss)

grad = 0
for i in range(1):
    grad += 2*(-x_data[i])*(y_data[i] - w.T@x_data[i])

print(grad)
