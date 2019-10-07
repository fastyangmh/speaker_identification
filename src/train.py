# import
import numpy as np
import torch
import torch.nn as nn
from glob import glob
from os.path import join, basename
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from model import *
from torch import optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score


#global parameters
USE_CUDA = torch.cuda.is_available()
SEED = 1234
np.random.seed(SEED)
torch.manual_seed(SEED)


# def
def load_npy(data_path, typ):
    files = glob(join(data_path, '*{}*'.format(typ)))
    for file in files:
        if '_data' in file:
            data = np.load(file)
        if '_label' in file:
            label = np.load(file)
    return data, label


def normailze(data, mean=None, std=None):
    mean = np.mean(data, 0) if mean is None else mean
    std = np.std(data, 0) if std is None else std
    data_std = (data-mean)/std
    return data_std, mean, std


def train_loop(dataloader, model, optimizer, criterion, epochs):
    train_loader, test_loader = dataloader
    history = []
    model.train()
    for epoch in tqdm(range(epochs)):
        total_loss = []
        for x, y in train_loader:
            if USE_CUDA:
                x, y = x.cuda(), y.cuda()
            yhat = model(x)
            loss = criterion(yhat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss.append(loss.item())
        history.append(np.mean(total_loss))
    return history


def evaluation(data, label, model, threshold=0.5):
    model.eval()
    data_tensor = torch.from_numpy(data).float()
    if USE_CUDA:
        data_tensor = data_tensor.cuda()
    prob = model(data_tensor).cpu().data.numpy()
    max_idx = np.argmax(prob, 1)
    pred = []
    for row, idx in enumerate(max_idx):
        if prob[row, idx] > threshold:
            pred.append(idx)
        else:
            pred.append(-1)
    pred = np.array(pred)
    acc = accuracy_score(np.argmax(label, 1), pred)
    return pred, acc, prob

# class


class normailze:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = np.mean(data, 0)
        self.std = np.std(data, 0)

    def transform(self, data):
        return (data-self.mean)/self.std


if __name__ == "__main__":
    # parameters
    data_path = './data'
    lr = 0.001
    batch_size = 128
    epochs = 40

    # load data
    typ = 'train'
    data, label = load_npy(data_path, typ)

    # preprocessing
    ohe = OneHotEncoder(sparse=False).fit(label.reshape(-1, 1))
    label_ohe = ohe.transform(label.reshape(-1, 1))
    x_train, x_test, y_train, y_test = train_test_split(
        data, label_ohe, test_size=0.3)
    stsc = normailze()
    stsc.fit(x_train)
    x_train_std = stsc.transform(x_train)
    x_test_std = stsc.transform(x_test)
    train_set = TensorDataset(torch.from_numpy(
        x_train_std).float(), torch.from_numpy(y_train).float())
    train_loader = DataLoader(dataset=train_set, batch_size=batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)

    # create model
    model = DCNN(180, 30, 30, 1, 0.5, out_channels=35)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    if USE_CUDA:
        model = model.cuda()

    # train
    history = train_loop((train_loader, None), model,
                         optimizer, criterion, epochs)
    # print(history)

    # evaluation
    pred_train, acc_train, prob_train = evaluation(x_train_std, y_train, model)
    pred_test, acc_test, prob_test = evaluation(x_test_std, y_test, model)
    print('Trainset accuracy: {}, imposter: {}'.format(
        *np.round((acc_train, (pred_train == -1).sum()/len(pred_train)), 4)))
    print('Testset accuracy: {}, imposter: {}'.format(
        *np.round((acc_test, (pred_test == -1).sum()/len(pred_test)), 4)))

    # load 3sec data
    data_3sec, label_3sec = load_npy(data_path, '3sec')
    label_3sec_ohe = ohe.transform(label_3sec.reshape(-1, 1))
    data_3sec_std = stsc.transform(data_3sec)
    pred_3sec, acc_3sec, prob_3sec = evaluation(
        data_3sec_std, label_3sec_ohe, model)
    print('3 second accuracy: {}, imposter: {}'.format(
        *np.round((acc_3sec, (pred_3sec == -1).sum()/len(pred_3sec)), 4)))

    # load 1sec data
    data_1sec, label_1sec = load_npy(data_path, '1sec')
    label_1sec_ohe = ohe.transform(label_1sec.reshape(-1, 1))
    data_1sec = np.append(data_1sec, np.zeros((896, 39, 120)), 2)
    data_1sec_std = stsc.transform(data_1sec)
    pred_1sec, acc_1sec, prob_1sec = evaluation(
        data_1sec_std, label_1sec_ohe, model)
    print('1 second accuracy: {}, imposter: {}'.format(
        *np.round((acc_1sec, (pred_1sec == -1).sum()/len(pred_1sec)), 4)))
