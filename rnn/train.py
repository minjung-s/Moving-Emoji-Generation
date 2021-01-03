import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim

import rnn_model

df = pd.read_csv('landmarks.csv', index_col=0)


arr = np.array(df)/480

X,y = arr[:,:136], arr[:,136:]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2, random_state = 0)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()
X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

train = TensorDataset(X_train, y_train)
test = TensorDataset(X_test, y_test)

train_loader = DataLoader(train, batch_size=4, shuffle=True)
test_loader = DataLoader(test, batch_size=4, shuffle=True)

# for i, data in enumerate(train_loader):
#     if i == 0:
#         t = data
#         break
#
# data[1].shape

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


def train_model(model, dataloaders, criterion, optimizer, num_epochs= 25):
    for epoch in range(num_epochs):

        for phase in ['train', 'test']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0

            for object,target in dataloaders[phase]:
                object = object.to(device)
                target = target.to(device)
                optimizer.zero_grad()

                outputs = model(object)
                outputs = outputs.view(object.size(0),-1)
                loss = criterion(outputs, target)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()*object.size(0)

            epoch_loss = running_loss/len(dataloaders[phase].dataset)
            print(f"[{epoch}/{num_epochs}][{phase}]loss={epoch_loss}")

    return model

model = rnn_model.one2many_lstm(t_stamp = 36).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = 0.0001)
dataloaders = {'train':train_loader, 'test':test_loader}

landmark_gen = train_model(model, dataloaders, criterion, optimizer, num_epochs=200)
landmark_gen.eval()
torch.save(landmark_gen,'generator.pytorch')
