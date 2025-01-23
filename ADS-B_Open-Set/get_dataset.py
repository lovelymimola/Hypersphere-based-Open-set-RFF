import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(num):
    x = np.load(f"/data/fuxue/Dataset_ADS-B/X_train_10Class.npy")
    y = np.load(f"/data/fuxue/Dataset_ADS-B/Y_train_10Class.npy")
    y = y.astype(np.uint8)
    train_map = {}
    index_all = []
    for i in range(0, num):
        index_i = [index for index, value in enumerate(y) if value == i]
        train_map[i] = x[index_i]
        index_all.append(index_i)
    index_all = np.concatenate(index_all)
    x_CloseSet = x[index_all]
    y_CloseSet = y[index_all]
    X_train, X_val, Y_train, Y_val = train_test_split(x_CloseSet, y_CloseSet, test_size=0.3, random_state=30)

    return X_train, X_val, Y_train, Y_val, train_map

def TestDataset(num, k):
    x = np.load(f"/data/fuxue/Dataset_ADS-B/X_test_10Class.npy")
    y = np.load(f"/data/fuxue/Dataset_ADS-B/Y_test_10Class.npy")
    y = y.astype(np.uint8)

    if k == 0:
        index_all = []
        for i in range(0, num):
            index_i = [index for index, value in enumerate(y) if value == i]
            index_all.append(index_i)
        index_all = np.concatenate(index_all)
        x_OpenSet = x[index_all]
        y_OpenSet = y[index_all]

    if k != 0:
        index_all = []
        for i in range(0, num):
            index_i = [index for index, value in enumerate(y) if value == i]
            index_all.append(index_i)
        for i in range(8, 8+k):
            index_i = [index for index, value in enumerate(y) if value == i]
            index_all.append(index_i)
        index_all = np.concatenate(index_all)
        x_OpenSet = x[index_all]
        y_OpenSet = y[index_all]

    test_map = {}
    unknown_test_map = {}
    for i in range(0, num):
        index_i = [index for index, value in enumerate(y_OpenSet) if value == i]
        test_map[i] = x_OpenSet[index_i]

    for i in range(8, 8+k):
        index_i = [index for index, value in enumerate(y_OpenSet) if value == i]
        unknown_test_map[i] = x_OpenSet[index_i]

    return x_OpenSet, y_OpenSet, test_map, unknown_test_map

if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val, train_map = TrainDataset(8)
