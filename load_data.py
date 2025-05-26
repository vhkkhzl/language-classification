# 1. Load data
import os
import glob
import pandas as pd
from tqdm.notebook import tqdm

# 設定資料夾路徑
data_folder = os.path.join("storage", "data", "language-words")
file_paths = glob.glob(os.path.join(data_folder, "*.txt"))
print(file_paths)

# 讀取所有語言的單詞
words_dict = {}
for file_path in file_paths:
    language = file_path.split("/")[3][:-4]
    with open(file_path, "r") as file:
        words = file.readlines()
        words = [word.strip().lower() for word in words]
        words_dict[language] = words
print(words_dict['German'][-10:])

# 顯示每個語言的單詞數量
for language, words in words_dict.items():
    print(language, len(words))
