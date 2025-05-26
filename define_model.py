import torch
import torch.nn as nn

# 設定模型參數
hidden_size = 24
num_layers = 2

device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = torch.device(device)
print(device)

# 定義 LSTM 模型
class Model(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc2 = nn.Linear(in_features=hidden_size, out_features=output_size)
    def forward(self, X):
        batch_size = X.size(0)
        h0 = torch.randn((self.num_layers, batch_size, self.hidden_size)).to(device)
        c0 = torch.randn((self.num_layers, batch_size, self.hidden_size)).to(device)
        out, ht = self.lstm1(X, (h0, c0))
        outn = out[:,-1,:]
        outn = outn.contiguous().view(batch_size, self.hidden_size)
        outn = self.fc2(outn)
        return outn

# 建立模型並移至裝置
model = Model(input_size=num_chars, output_size=num_langs, hidden_size=hidden_size, num_layers=num_layers)
model = nn.DataParallel(model)
model = model.to(device)
print(model)

# 檢查模型參數型態
for p in model.parameters():
    print(p.dtype)

# (可選) #summary(model, input_size=(max_timesteps, num_chars))
# (可選) #writer.add_graph(model, X)
# (可選) #writer.close()
