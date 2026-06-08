import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# data=pd.read_csv(r'medical_data_logi.csv')
# x_train=data['fasting_glucose_mg_dl']
# y_train=data['diabetes_label']

x_train=np.arange(10,dtype='float32').reshape((10,1))
y_train=np.array([1.0,1.3,3.1,2.0,5.0,6.3,6.6,7.4,8.0,9.0],dtype='float32')

x_train=torch.from_numpy(x_train)
y_train=torch.from_numpy(y_train)
train_ds=TensorDataset(x_train,y_train)
train_dl=DataLoader(train_ds,batch_size=2,shuffle=True)

torch.manual_seed(1)
weight=torch.randn(1,1,requires_grad=True)
bias=torch.zeros(1,1,requires_grad=True)
def model(xb):
    return xb @ weight + bias

def loss_fn(input,target):
    return (input-target).pow(2).mean()

optimizer=torch.optim.SGD([weight,bias],lr=0.001)
num_epochs=100

def train(train_dl,loss_fn,optimizer):
    for epoch in range(num_epochs):
        total=0
        for x_batch,y_batch in tqdm(train_dl):
            pred=model(x_batch)
            loss=loss_fn(pred,y_batch.unsqueeze(1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        total = total/len(train_dl)
        print(f'epoch {epoch}, Loss: {total:.4f} ')

train(train_dl,loss_fn,optimizer)

print('Weights:',weight.item())
print('Bias:',bias.item())

sort_idx=torch.argsort(x_train.squeeze())
x_sort=x_train[sort_idx]

with torch.no_grad():
    y_sort=model(x_sort)

plt.scatter(x_train.numpy(),y_train.numpy())
plt.plot(x_sort.numpy(),y_sort.numpy())

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Linear Regression")
plt.show()

