import torch
from torch.utils.data import Dataset

# 定義自訂的 Dataset
class WordDataset(Dataset):
    def __init__(self, words_df):
        self.words_df = words_df
    def __len__(self):
        self.len = len(self.words_df)
        return self.len
    def __getitem__(self, idx):
        row = self.words_df.iloc[idx, :]
        word = row['word'].ljust(max_timesteps)
        x = torch.zeros((max_timesteps, num_chars))  # [timesteps, input_size]
        for i, char in enumerate(word):
            x[i, char_to_id[char]] = 1
        y = lang_to_id[row['language']]
        return x, y

# 建立訓練集與測試集的 Dataset
train_set = WordDataset(words_df_train)
test_set = WordDataset(words_df_test)
