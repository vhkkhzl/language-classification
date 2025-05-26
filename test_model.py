import os
import torch
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 載入模型
path = os.path.join("storage", "models", "language-words", "classifier.pth")
model = Model(input_size=num_chars, output_size=num_langs, hidden_size=hidden_size, num_layers=num_layers)
model = nn.DataParallel(model)
model.load_state_dict(torch.load(path))
# model = model.to("cpu")

# 訓練集預測
with torch.no_grad():
    Y_train, Y_pred_train = [], []
    for X_mb, Y_mb in tqdm(train_loader):
        out = model(X_mb)
        _, Y_pred_mb = torch.max(out, 1)
        Y_train.extend(Y_mb.numpy().tolist())
        Y_pred_train.extend(Y_pred_mb.cpu().numpy().tolist())

# 測試集預測
with torch.no_grad():
    Y_test, Y_pred_test = [], []
    for X_mb, Y_mb in tqdm(test_loader):
        out = model(X_mb)
        _, Y_pred_mb = torch.max(out, 1)
        Y_test.extend(Y_mb.numpy().tolist())
        Y_pred_test.extend(Y_pred_mb.cpu().numpy().tolist())

# 計算準確率
train_accuracy = accuracy_score(Y_train, Y_pred_train)
test_accuracy = accuracy_score(Y_test, Y_pred_test)
print("Train Accuracy: {}".format(train_accuracy))
print("Test Accuracy: {}".format(test_accuracy))

# 混淆矩陣
labels = sorted(list(lang_to_id.keys()))
c_mat_train = confusion_matrix(Y_train, Y_pred_train)
c_mat_train = c_mat_train / c_mat_train.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(15,5))
sns.heatmap(c_mat_train, annot=True, fmt="0.2f", xticklabels=labels, yticklabels=labels)
plt.title('Train Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

c_mat_test = confusion_matrix(Y_test, Y_pred_test)
c_mat_test = c_mat_test / c_mat_test.sum(axis=1)[:, np.newaxis]
plt.figure(figsize=(15, 5))
sns.heatmap(c_mat_test, annot=True, fmt='0.2f', xticklabels=labels, yticklabels=labels)
plt.title('Test Data')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

def compute_accuracies(c_mat):
    accuracies = c_mat.astype('float') / c_mat.sum(axis=1)
    accuracies = accuracies.diagonal()
    accuracies = {k:v for k, v in zip(labels, accuracies)}
    return accuracies

print(compute_accuracies(c_mat_train))
print(compute_accuracies(c_mat_test))
