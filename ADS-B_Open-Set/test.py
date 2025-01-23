import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
from models import ConvAngularPen, ConvBaseline
from get_dataset import *
import sklearn.metrics as sm
import scipy.io
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed) # CPU
    torch.cuda.manual_seed(seed) # GPU
    torch.cuda.manual_seed_all(seed) # All GPU
    os.environ['PYTHONHASHSEED'] = str(seed) # 禁止hash随机化
    torch.backends.cudnn.deterministic = True # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False # True的话会自动寻找最适合当前配置的高效算法，来达到优化运行效率的问题。False保证实验结果可复现

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

def test_am_openset(test_loader, model):
    model = model.to(device).eval()
    out_probs_all = []
    labels_all = []
    feat_embed_all = []
    with torch.no_grad():
        for i, (feats, labels) in enumerate(test_loader):
            feats = feats.to(device)
            labels = labels.long().to(device)
            feat_embed = model.convlayers(feats)
            feat_embed_norm = F.normalize(feat_embed, p=2, dim=1)
            out = model.adms_loss.fc(feat_embed_norm)
            out_probs = F.softmax(out, dim=1)
            feat_embed_all.append(feat_embed_norm.detach().cpu().numpy())
            out_probs_all.append(out_probs.detach().cpu().numpy())
            labels_all.append(labels.detach().cpu().numpy())
    return np.concatenate(feat_embed_all), np.concatenate(out_probs_all), np.concatenate(labels_all)

def Data_prepared(n_classes):
    X_train, X_val, value_Y_train, value_Y_val, train_map = TrainDataset(n_classes)

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
    X_train, X_val, value_Y_train, value_Y_val, train_map = TrainDataset(n_classes)

    max_value, min_value = Data_prepared(n_classes)

    X_train = (X_train - min_value) / (max_value - min_value)
    X_val = (X_val - min_value) / (max_value - min_value)

    X_train = X_train.transpose(0, 2, 1)
    X_val = X_val.transpose(0, 2, 1)

    return X_train, X_val, value_Y_train, value_Y_val

def TestDataset_prepared(num, k):
    X_test, value_Y_test, test_map, unknown_test_map = TestDataset(num, k)

    max_value, min_value = Data_prepared(num)

    X_test = (X_test - min_value) / (max_value - min_value)
    X_test = X_test.transpose(0, 2, 1)

    return X_test, value_Y_test

def parse_args():
    parser = argparse.ArgumentParser(description='Run Angular Penalty and Baseline experiments in fMNIST')
    parser.add_argument('--n_classes', type=int, default=8,
                        help='input batch size for training (default: 10)')
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
    X_train, X_val, Y_train, Y_val = TrainDataset_prepared(args.n_classes)
    X_test, Y_test = TestDataset_prepared(args.n_classes, 2)

    train_dataset = TensorDataset(torch.Tensor(X_train), torch.Tensor(Y_train))
    test_OpenSet_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(Y_test))

    example_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=False)
    test_OpenSet_loader = torch.utils.data.DataLoader(dataset=test_OpenSet_dataset, batch_size=args.batch_size, shuffle=False)

    loss_types = ['cosface']
    for loss_type in loss_types:
        model_save_path = 'model_weight/' + loss_type + 'm02_model_r8.pth'
        model_am = torch.load(model_save_path)
        print(model_am)
        am_embeds_norm, am_labels = get_embeds(model_am, example_loader)
        scipy.io.savemat('./EVT/data/' + loss_type + '_m02_r8_classes8.mat', mdict={'am_embeds_norm': am_embeds_norm, 'am_labels': am_labels})

        feat_embed_open, out_prob_open, label_open = test_am_openset(test_OpenSet_loader, model_am)
        scipy.io.savemat('./EVT/data/test_'+ loss_type + '_m02_r8_classes8+2.mat',mdict={'test_am_embeds_norm': feat_embed_open, 'test_out_prob_open': out_prob_open, 'test_am_labels': label_open})


if __name__ == "__main__":
    args = parse_args()
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    set_seed(args.seed)
    main()
