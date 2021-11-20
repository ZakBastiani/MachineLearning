import numpy as np

x = np.array([[0.5, -1, 0.3, 1],
              [-1, -2, -2, 1],
              [1.5, 0.2, -2.5, 1]])
y = np.array([1, -1, 1])
w = np.zeros(4)
N = 3
C = 1/3
lr_list = [0.01, 0.005, 0.0025]

for j in range(len(x)):
    lr = lr_list[j]
    if y[j] * w.T@x[j] <= 1:
        grad = -lr * np.append(w[:len(w)-1], 0) \
               + lr * C * N * y[j] * x[j]
        w = w + grad
    else:
        w[:len(w)-1] = (1-lr)*w[:len(w)-1]  # don't change the bias
    print(grad)




