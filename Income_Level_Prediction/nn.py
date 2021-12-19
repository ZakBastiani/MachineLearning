import numpy as np
import pandas as pd


class NN:
    def __init__(self, layer_size, dim):
        self.layer_size = layer_size
        self.dim = dim
        self.x = np.zeros((1, dim+1))
        self.weights_1 = np.random.normal(0, 0.5, (dim+1, layer_size-1))
        self.layer1_zs = np.zeros((1, layer_size-1))
        self.weights_2 = np.random.normal(0, 0.5, (layer_size, layer_size-1))
        self.layer2_zs = np.zeros((1, layer_size-1))
        self.weights_3 = np.random.normal(0, 0.5, (layer_size, 1))
        self.y = 0

    def train(self, dataset, lr_0, d):
        t = 1
        for T in range(50):
            rand_data = dataset.sample(len(dataset), replace=False, ignore_index=True)
            for j in range(len(rand_data)):
                lr = lr_0/(1+(lr_0/d)*t)
                x = rand_data.drop(columns=['label']).iloc[j].to_numpy()
                y = rand_data['label'].iloc[j]
                self.forward(x)
                self.backward(y, lr)
                t += 1
        print('Done Training')

    def test(self, dataset):
        ans = np.zeros(len(dataset))
        for i in range(len(dataset)):
            ans[i] = self.forward(dataset.iloc[i].to_numpy())
        return ans

    def forward(self, data):
        self.x = np.insert(data, [0], 1)
        self.layer1_zs = self.sigma(self.x @ self.weights_1)
        self.layer1_zs = np.insert(self.layer1_zs, [0], 1)
        self.layer2_zs = self.sigma(self.layer1_zs @ self.weights_2)
        self.layer2_zs = np.insert(self.layer2_zs, [0], 1)
        self.y = (self.layer2_zs @ self.weights_3)[0]
        return self.y

    def backward(self, obs_y, lr):
        dy = self.y - obs_y
        short_extend_w3 = np.repeat(self.weights_3[1:], self.layer_size, axis=1).T
        short_extend_z2 = np.repeat(np.expand_dims(self.layer2_zs.T[1:], axis=1), self.layer_size, axis=1).T
        full_extend_z1 = np.repeat(np.expand_dims(self.layer1_zs, axis=1), self.layer_size-1, axis=1)
        short_extend_z1 = np.repeat(np.expand_dims(self.layer1_zs[1:], axis=1), self.dim+1, axis=1).T
        mid_extend_z2 = np.repeat(np.expand_dims(self.layer2_zs.T[1:], axis=1), self.layer_size-1, axis=1)
        extend_data = np.repeat(np.expand_dims(self.x, axis=1), self.layer_size-1, axis=1)

        middle_chunk = np.repeat(self.weights_3[1:], self.dim+1, axis=1).T @ (mid_extend_z2 * (1-mid_extend_z2) * self.weights_2[1:].T)

        self.weights_3 = self.weights_3 - lr * dy * np.expand_dims(self.layer2_zs, axis=1)
        self.weights_2 = self.weights_2 - lr * dy * short_extend_w3 * short_extend_z2 * (1 - short_extend_z2) * full_extend_z1
        self.weights_1 = self.weights_1 - lr * dy * middle_chunk * short_extend_z1 * (1 - short_extend_z1) * extend_data

    def sigma(self, num):
        return 1/(1+np.exp(-num))
