import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models import ConvAngularPen
from get_dataset import *
import sklearn.metrics as sm
import scipy.io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

def train_am(train_loader, val_loader, loss_type, model_save_path, writer):
    model = ConvAngularPen(loss_type=loss_type).to(device)
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    current_val_loss = 100
    for epoch in range(args.num_epochs):
        model.train()
        train_loss_all = train_correct = 0
        for i, (feats, labels) in enumerate(train_loader):
            feats = feats.to(device)
            labels = labels.long().to(device)
            loss, logit = model(feats, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_all += loss.item()
            out_probs = F.softmax(logit, dim=1)
            pred = out_probs.argmax(dim=1, keepdim=True)
            train_correct += pred.eq(labels.view_as(pred)).sum().item()
        train_loss_all /= len(train_loader)
        print(f"-----------------------------------Epoch: {epoch}---------------------------------------")
        print('Train \tClassifier_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
            train_loss_all,
            train_correct,
            len(train_loader.dataset),
            100.0 * train_correct / len(train_loader.dataset))
        )

        model.eval()
        val_loss_all = val_correct = 0
        with torch.no_grad():
            for i, (feats, labels) in enumerate(val_loader):
                feats = feats.to(device)
                labels = labels.long().to(device)
                loss, logit = model(feats, labels=labels)
                val_loss_all += loss.item()
                out_probs = F.softmax(logit, dim=1)
                pred = out_probs.argmax(dim=1, keepdim=True)
                val_correct += pred.eq(labels.view_as(pred)).sum().item()
            val_loss_all /= len(val_loader)

            print('Val \tClassifier_Loss: {:.6f}, Accuracy: {}/{} ({:0f}%)\n'.format(
                val_loss_all,
                val_correct,
                len(val_loader.dataset),
                100.0 * val_correct / len(val_loader.dataset))
            )
            if current_val_loss > val_loss_all:
                print(f"the val loss decreases from {current_val_loss} to {val_loss_all}, the new model is saved.")
                current_val_loss = val_loss_all
                torch.save(model, model_save_path)

def get_embeds(model, loader):
    model = model.to(device).eval()
    full_embeds_normalize = []
    full_labels = []
    with torch.no_grad():
        for i, (feats, labels) in enumerate(loader):
            feats = feats.to(device)
            embeds = model(feats, embed=True)
            full_embeds_normalize.append(F.normalize(embeds, p=2, dim=1).detach().cpu().numpy())
            full_labels.append(labels.cpu().detach().numpy())
    return np.concatenate(full_embeds_normalize), np.concatenate(full_labels)

def micro_f_measure(label, predict):
    tn = fp = tp = fn = 0
    for index in range(label.shape[0]):
        if label[index] == -1:
            if label[index] == predict[index]:
                tn = tn + 1
            else:
                fp = fp + 1
        else:
            if label[index] == predict[index]:
                tp = tp + 1
            else:
                fn = fn + 1

    prcn = tp/(tp+fp)
    rcll = tp/(tp+fn)
    f = 2 * ((prcn*rcll)/(prcn+rcll))

    return f

def test_am(test_loader, model):
    model = model.to(device).eval()
    correct = 0
    with torch.no_grad():
        for i, (feats, labels) in enumerate(test_loader):
            feats = feats.to(device)
            labels = labels.long().to(device)
            loss, out = model(feats, labels)
            out_probs = F.softmax(out, dim=1)
            pred = out_probs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()
        print('Test \tAccuracy: {}/{} ({:0f}%)\n'.format(
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset))
        )

def Data_prepared(n_classes):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes)

    min_value = X_train.min()
    min_in_val = X_val.min()
    if min_in_val < min_value:
        min_value = min_in_val

    max_value = X_train.max()
    max_in_val = X_val.max()
    if max_in_val > max_value:
        max_value = max_in_val

    return max_value, min_value

def TrainDataset_prepared(n_classes):
    X_train, X_val, value_Y_train, value_Y_val = TrainDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes)

    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return X_train, X_val, value_Y_train, value_Y_val

def TestDataset_prepared(n_classes):
    X_test, value_Y_test = TestDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes)

    X_test = (X_test - min_value) / (max_value - min_value)

    X_test = X_test.transpose(0, 2, 1)

    return X_test, value_Y_test

def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Penalty and Baseline experiments in ADS-B')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument('--num-epochs', type=int, default=300,
                        help='Number of epochs to train each model for (default: 20)')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed (default: 1234)')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='learning rate (default: 0.001)')
    parser.add_argument('--use-cuda', action='store_true', default=True,
                        help='enables CUDA training')
    args = parser.parse_args()

    return args

def main():
    X_train, X_val, Y_train, Y_val = TrainDataset_prepared(8)
    X_test, Y_test = TestDataset_prepared(8)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    val_dataset = TensorDataset(torch.Tensor(X_val), torch.Tensor(Y_val))
    test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,shuffle=True)
    val_loader = torch.utils.data.DataLoader(dataset=val_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=True)

    writer = SummaryWriter("logs")
    loss_types = ['cosface']
    for loss_type in loss_types:
        print('Training {} model....'.format(loss_type))
        model_save_path = 'model_weight/' + loss_type + 'm02_model_r8.pth'
        train_am(train_loader, val_loader, loss_type, model_save_path, writer)
        model_am = torch.load(model_save_path)
        test_am(val_loader, model_am)

if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(args.seed)
    main()
