import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(num):
    x = np.load(f"D:/Dataset/ADS-B_4800/X_train_10Class.npy")
    y = np.load(f"D:/Dataset/ADS-B_4800/Y_train_10Class.npy")
    y = y.astype(np.uint8)
    index_all = []
    for i in range(0,num):
        index_i = [index for index, value in enumerate(y) if value == i]
        index_all.append(index_i)
    index_all = np.concatenate(index_all)
    x_CloseSet = x[index_all]
    y_CloseSet = y[index_all]
    X_train, X_val, Y_train, Y_val = train_test_split(x_CloseSet, y_CloseSet, test_size=0.3, random_state=30)

    return X_train, X_val, Y_train, Y_val

def TestDataset(num):
    x = np.load(f"D:/Dataset/ADS-B_4800/X_test_10Class.npy")
    y = np.load(f"D:/Dataset/ADS-B_4800/Y_test_10Class.npy")
    y = y.astype(np.uint8)
    index_all = []
    for i in range(0,num):
        index_i = [index for index, value in enumerate(y) if value == i]
        index_all.append(index_i)
    index_all = np.concatenate(index_all)
    x_CloseSet = x[index_all]
    y_CloseSet = y[index_all]

    return x_CloseSet, y_CloseSet

if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = TrainDataset(8)

    x_CloseSet, y_CloseSet = TestDataset(8)