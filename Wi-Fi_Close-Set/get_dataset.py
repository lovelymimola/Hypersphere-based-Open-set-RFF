import torch
import numpy as np
from sklearn.model_selection import train_test_split
import random

def TrainDataset(num):
    x = np.load(f"D:/Dataset/16WiFi/IoT_WiFi/归一化1（两篇IoTJ采用）/run1/Feet62_X_train.npy")
    y = np.load(f"D:/Dataset/16WiFi/IoT_WiFi/归一化1（两篇IoTJ采用）/run1/Feet62_Y_train.npy")
    y = y.astype(np.uint8)
    index_all = []
    for i in range(0, num):
        index_i = [index for index, value in enumerate(y) if value == i]
        index_all.append(index_i)
    index_all = np.concatenate(index_all)
    x_CloseSet = x[index_all]
    y_CloseSet = y[index_all]
    X_train, X_val, Y_train, Y_val = train_test_split(x_CloseSet, y_CloseSet, test_size=0.3, random_state=30)

    return X_train, X_val, Y_train, Y_val

def TestDataset(num):
    x = np.load(f"D:/Dataset/16WiFi/IoT_WiFi/归一化1（两篇IoTJ采用）/run1/Feet62_X_test.npy")
    y = np.load(f"D:/Dataset/16WiFi/IoT_WiFi/归一化1（两篇IoTJ采用）/run1/Feet62_Y_test.npy")
    y = y.astype(np.uint8)

    index_all = []
    for i in range(0, num):
        index_i = [index for index, value in enumerate(y) if value == i]
        index_all.append(index_i)
    index_all = np.concatenate(index_all)
    x_CloseSet = x[index_all]
    y_CloseSet = y[index_all]
    return x_CloseSet, y_CloseSet

if __name__ == "__main__":
    X_train, X_val, Y_train, Y_val = TrainDataset(10)

    x_CloseSet, y_CloseSet = TestDataset(16)