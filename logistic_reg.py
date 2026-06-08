import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

data = pd.read_csv('medical_data_logi.csv')

x = data['fasting_glucose_mg_dl'].values.astype(np.float32).reshape(-1,1)
y = data['diabetes_label'].values.astype(np.float32).reshape(-1,1)

x = (x - x.mean()) / x.std()

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train)
x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test)

train_ds = TensorDataset(x_train, y_train)
test_ds = TensorDataset(x_test, y_test)

train_dl = DataLoader(train_ds, batch_size=16, shuffle=True)
test_dl = DataLoader(test_ds, batch_size=16)

torch.manual_seed(1)

weight = torch.randn(1,1, requires_grad=True)
bias = torch.zeros(1, requires_grad=True)

def model(x):
    linear = x @ weight + bias
    return torch.sigmoid(linear)

def loss_fn(y_pred, y):
    eps = 1e-8
    return -torch.mean(
        y * torch.log(y_pred + eps) +
        (1 - y) * torch.log(1 - y_pred + eps)
    )

optimizer = torch.optim.SGD([weight, bias], lr=0.01)

for epoch in range(100):
    train_loss = 0

    for x_batch, y_batch in train_dl:
        pred = model(x_batch)
        loss = loss_fn(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_dl)

    with torch.no_grad():
        correct = 0
        total = 0

        for x_batch, y_batch in test_dl:
            pred = model(x_batch)
            pred_class = (pred >= 0.5).float()

            correct += (pred_class == y_batch).sum().item()
            total += y_batch.size(0)

        accuracy = correct / total

    print(f"Epoch:{epoch}, Loss:{train_loss:.4f}, Accuracy:{accuracy:.4f}")

print("Weight:", weight.item())
print("Bias:", bias.item())

x_test_np = x_test.numpy()
y_test_np = y_test.numpy()

sorted_idx = x_test_np[:, 0].argsort()
x_sorted = x_test_np[sorted_idx]

with torch.no_grad():
    y_pred_sorted = model(torch.from_numpy(x_sorted)).numpy()

plt.scatter(x_test_np, y_test_np, alpha=0.6)
plt.plot(x_sorted, y_pred_sorted, linewidth=3)

plt.xlabel("Scaled Fasting Glucose")
plt.ylabel("Probability of Diabetes")
plt.title("Logistic Regression Fit")
plt.show()
