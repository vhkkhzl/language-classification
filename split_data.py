import pandas as pd
from tqdm.notebook import tqdm
from sklearn.model_selection import train_test_split
from load_data import words_dict

# 將 words_dict 轉為 DataFrame
words_df = []
for language, words in tqdm(words_dict.items()):
    for word in words:
        words_df.append({"word": word, "language": language})
words_df = pd.DataFrame(words_df)
print(words_df.shape)
words_df.head()

# 切分訓練集與測試集
words_df_train, words_df_test = train_test_split(words_df, train_size=0.8, stratify=words_df["language"], random_state=0)
words_df_train = words_df_train.reset_index(drop=True)
words_df_test = words_df_test.reset_index(drop=True)
print(words_df_train.shape)
print(words_df_test.shape)

# 統計各語言在訓練集與測試集的數量
train_count = words_df_train["language"].value_counts().rename("Train")
test_count = words_df_test["language"].value_counts().rename("Test")
count = pd.concat([train_count, test_count], axis=1, sort=True).T
count.loc["Total", :] = count.sum(axis=0)  # add row
count.loc[:, "Total"] = count.sum(axis=1)  # add col
count = count.astype("int")
print(count)
