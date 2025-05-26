# 2. Clean text
import unicodedata
import string
import copy

# 取得所有出現過的字元
characters = set()
for language, words in words_dict.items():
    for word in words:
        characters.update(list(word))
characters = sorted(list(characters))
print(characters)

# 範例：重音字元正規化
c = list('ñ')
c_normalised = list(unicodedata.normalize("NFD", 'ñ'))
print(c, c_normalised)

# 將所有字元正規化
characters_normalised = []
for character in characters:
    character_normalised = unicodedata.normalize("NFD", character)[0]
    characters_normalised.append(character_normalised)
print(characters_normalised)

# 定義可接受的字元
characters_all = list(string.ascii_lowercase + " -',:;")
print(len(characters_all), characters_all)

def clean_word(word):
    cleaned_word = ""
    for character in word:
        for character_raw in unicodedata.normalize('NFD', character):
            if character_raw in characters_all:
                cleaned_word += character_raw
    return cleaned_word

# 清理所有單詞
words_dict_cleaned = {}
for language, words in words_dict.items():
    cleaned_words = []
    for word in words:
        cleaned_word = clean_word(word)
        cleaned_words.append(cleaned_word)
    words_dict_cleaned[language] = cleaned_words

# 檢查清理前後的差異
print(words_dict['German'][-10:])
print(words_dict_cleaned['German'][-10:])
print(words_dict['Portuguese'][-10:])
print(words_dict_cleaned['Portuguese'][-10:])
print(words_dict['Polish'][-10:])
print(words_dict_cleaned['Polish'][-10:])

# 用清理後的資料取代原本的資料
words_dict = copy.deepcopy(words_dict_cleaned)
del words_dict_cleaned
