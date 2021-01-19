import pandas as pd
import numpy as np
import copy

from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import conditional_rnn_model


def train_model(model, dataloaders, criterion, optimizer, num_epochs, device):
    global best_loss

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1.0

    for epoch in range(num_epochs):
        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for object, target in dataloaders[phase]:
                object = object.to(device)
                object, condition = object[:,:136], object[:,136]
                target = target.to(device)
                optimizer.zero_grad()

                outputs = model(object, condition)
                loss = criterion(outputs, target)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()*object.size(0)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            print(f"[{epoch}/{num_epochs}][{phase}]loss={epoch_loss}")

            if phase == 'test' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    return model


def trainer(data_path, t_stamp = 50, batch = 64, num_epochs = 1000, conditions = 'element_mul'):
    df = pd.read_csv(data_path, index_col = 0)
    arr, condition = df.iloc[:,:(t_stamp+1)*136], df['condition']
    del df

    arr = np.array(arr)/1024
    arr = np.exp(arr)
    condition = np.array(condition)

    X, y = arr[:,:136], arr[:,136:]
    condition = condition.reshape(X.shape[0],1)
    X = np.concatenate((X, condition), axis = 1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()

    train = TensorDataset(X_train, y_train)
    test = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train, batch_size=batch, shuffle=True)
    test_loader = DataLoader(test, batch_size=batch, shuffle=True)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    model = conditional_rnn_model.conditional_rnn(t_stamp = t_stamp, conditions = conditions).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr = 0.0001)
    dataloaders = {'train': train_loader, 'test': test_loader}

    landmark_gen = train_model(model, dataloaders, criterion, optimizer, num_epochs, device)

    return landmark_gen
