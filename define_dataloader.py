import os
from torch.utils.data import DataLoader
from define_dataset import *

# 設定 batch size
train_batch_size = 128
test_batch_size = 4

# 取得 CPU 數量
num_cpus = os.cpu_count()
print(num_cpus)

# 建立 DataLoader
train_loader = DataLoader(train_set, batch_size=train_batch_size, shuffle=True, num_workers=num_cpus)
test_loader = DataLoader(test_set, batch_size=test_batch_size, shuffle=False, num_workers=num_cpus)

# 檢查 DataLoader 輸出
train_iter = iter(train_loader)
X, Y = train_iter.next()
print(X.size(), Y.size())

# 訓練集的 batch 數量
len_train_loader = len(train_loader)
print(len_train_loader)
