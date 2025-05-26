from load_data import words_dict
from clean_text import characters_all

num_langs = len(words_dict.keys())
num_chars = len(characters_all)
print(num_langs, num_chars)

# 找出最長單詞長度
max_timesteps = 0
for language, words in words_dict.items():
    for word in words:
        if len(word) > max_timesteps:
            max_timesteps = len(word)
print(max_timesteps)

# 建立語言與索引的對應
lang_to_id = {k:v for k, v in zip(sorted(list(words_dict.keys())), range(len(words_dict.keys())))}
print(lang_to_id)

id_to_lang = {v:k for k, v in lang_to_id.items()}
print(id_to_lang)

# 建立字元與索引的對應
char_to_id = {k:v for k, v in zip(characters_all, range(len(characters_all)))}
print(char_to_id)

id_to_char = {v:k for k, v in char_to_id.items()}
print(id_to_char)
