import torch
import torch.nn as nn
import torch.optim as optim
from define_dataloader import *
from define_model import model
from test_model import device

# 學習率與 scheduler 設定
lr = 0.01
step_size = len_train_loader * 4
gamma = 0.95
print(step_size)

# 計算類別權重
alpha = 0.6
weights = len(words_df_train) / (words_df_train['language'].value_counts() ** alpha)
weights = weights / weights.sum()
weights = weights.sort_index()
print(weights)

weights = torch.Tensor(weights).to(device)
print(weights)

# 定義損失函數、優化器與學習率調整器
criterion = nn.CrossEntropyLoss(weight=weights, reduction="mean")
optimizer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size, gamma=gamma)
