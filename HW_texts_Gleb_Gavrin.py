#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from typing import  List
import matplotlib.pyplot as plt
import seaborn as sns
from string import punctuation


# В этом домашнем задании вам предстоит построить классификатор текстов.
# 
# Будем предсказывать эмоциональную окраску твиттов о коронавирусе.

# Для каждого твитта указано:
# 
# 
# *   UserName - имя пользователя, заменено на целое число для анонимности
# *   ScreenName - отображающееся имя пользователя, заменено на целое число для анонимности
# *   Location - местоположение
# *   TweetAt - дата создания твитта
# *   OriginalTweet - текст твитта
# *   Sentiment - эмоциональная окраска твитта (целевая переменная)

# # Задание 1 Подготовка (0.5 балла)

# In[2]:


df = pd.read_csv('tweets_coronavirus.csv')
df.sample(4)


# In[3]:


df.Sentiment.unique()


# In[4]:


df['Emotion'] = [1 if x == 'Positive' or x == 'Extremely Positive' else 0 for x in df['Sentiment']]


# In[5]:


df.head()


# In[6]:


df.drop('Sentiment', axis= 1 , inplace= True )


# In[7]:


df.head()


# In[8]:


df.rename(columns={"Emotion": "Emotion"})


# Преобразовали целевую переменную в бинарный вид

# In[9]:


list(df.columns)


# In[10]:


df['Emotion'].value_counts()


# In[11]:


15398/33443


# Присутствует дисбаланс классов, но он незначительный

# In[12]:


#проверяем на пропуски
df.isnull().mean()


# In[13]:


#пропуски только в столбце Location. Заполним их Unknown:


# In[14]:


df.fillna({'Location':' Unknown'}, inplace= True )


# In[15]:


df.head()


# In[16]:


from sklearn.model_selection import train_test_split

#X = df.iloc[:, :-1]
#y = df.iloc[:, -1]
 

train, test = train_test_split(df, test_size=0.3, random_state=0)


# In[17]:


train.shape


# In[18]:


test.shape


# ## Задание 2 Токенизация (3 балла)

# Постройте словарь на основе обучающей выборки и посчитайте количество встреч каждого токена с использованием самой простой токенизации - деления текстов по пробельным символам и приведение токенов в нижний регистр

# In[19]:


text1 = ' '.join(train['OriginalTweet'].tolist())


# In[20]:


#text1


# In[21]:


from itertools import chain
from collections import Counter
from nltk.tokenize import sent_tokenize, word_tokenize

wordlist = list(chain(*[word_tokenize(s) for s in sent_tokenize(text1)]))
print(Counter(wordlist))


# In[22]:


#размер словаря
len(Counter(wordlist))


# Выведите 10 самых популярных токенов с количеством встреч каждого из них. Объясните, почему именно эти токены в топе

# In[23]:


sorted_dict1 = sorted(Counter(wordlist).items(), key=lambda kv: kv[1], reverse=True)


# In[24]:


sorted_dict1


# In[25]:


sorted_dict1[:10]


# Именно эти токены в топе потому, что это знаки, предлоги. Они будут в топе почти в любом тексте. Если мы хотим увидеть содержательный топ слов, стоит смотреть дальше первых 10.

# In[26]:


sorted_dict1[11:35]


# Здесь уже встречаются ожидаемые слова(coronavirus, prices, COVID-19 и др.)

# Удалите стоп-слова из словаря и выведите новый топ-10 токенов (и количество встреч) по популярности. Что можно сказать о нем?

# In[27]:


import nltk
from nltk.corpus import stopwords


# In[28]:


stops = set(stopwords.words('english'))
print(stops)


# In[29]:


#text = "NLTK помогает в удалении стоп-слов из текста."
#tokens = word_tokenize(text1)
#stop_words = set(stopwords.words('english'))
#filtered_tokens = [word for word in tokens if word not in stops]

#print(filtered_tokens)


# In[30]:


dict1_without_stops = []
for word0 in sorted_dict1:
    if word0[0] not in stops:
        dict1_without_stops.append(word0)
        


# In[31]:


dict1_without_stops


# In[32]:


len(dict1_without_stops)


# In[33]:


dict1_without_stops = sorted(dict1_without_stops, key=lambda kv: kv[1], reverse=True)


# In[34]:


dict1_without_stops[:10] 


# теперь больше пунктуационных знаков в топе, слова типа the ушли

# Также выведите 20 самых непопулярных слов (если самых непопулярных слов больше выведите любые 20 из них) Почему эти токены непопулярны, требуется ли как-то дополнительно работать с ними?

# In[35]:


#теперь выведем 20 самых непопулярных слов
dict1_without_stops[-20:]


# Здесь в основном никнеймы/названия/ссылки, на них можно не обращать внимания

# Теперь воспользуемся токенайзером получше - TweetTokenizer из библиотеки nltk. Примените его и посмотрите на топ-10 популярных слов. Чем он отличается от топа, который получался раньше? Почему?

# In[36]:


from nltk.tokenize import TweetTokenizer


# In[37]:


tw = TweetTokenizer()
tw.tokenize(text1)


# In[38]:


counts_Tweet = Counter(tw.tokenize(text1))
counts_Tweet


# Отсортируем этот словарь по убыванию:

# In[39]:


counts_Tweet = sorted(counts_Tweet.items(), key=lambda kv: kv[1], reverse=True)


# In[40]:


counts_Tweet[:10]


# топ-10 похож на топ из первой токенизации, но больше предлогов

# Это обусловлено тем, что TweetTokenizer по другому обрабатывает смайлики, хештеги. Это удобно для анализа твитов.

# Удалите из словаря стоп-слова и пунктуацию, посмотрите на новый топ-10 слов с количеством встреч, есть ли теперь в нем что-то не похожее на слова?

# In[41]:


from string import punctuation


# In[42]:


for pair in counts_Tweet:
    if pair[0] in punctuation:
        counts_Tweet.remove(pair)


# In[43]:


cTweet_without_stops = []
for wordT in counts_Tweet:
    if wordT[0] not in stops:
        cTweet_without_stops.append(wordT)


# In[44]:


cTweet_without_stops[:10]


# Топ-10 слов стал более анализируемым. Теперь здесь боьшинство слов связаны с пандемией и карантином

# Удалите из словаря токены из одного символа, с позицией в таблице Unicode 128 и более (ord(x) >= 128)
# 
# Выведите топ-10 самых популярных и топ-20 непопулярных слов. Чем полученные топы отличаются от итоговых топов, полученных при использовании токенизации по пробелам? Что теперь лучше, а что хуже?

# Теперь удалим токены из 1 символа и токены ord>=128:

# In[45]:


cTweet_without_stops_128 = []
for word128 in cTweet_without_stops:
    if word128[0].isascii() == True and len(word128[0]) != 1 :
        cTweet_without_stops_128.append(word128)
        


# In[46]:


len(cTweet_without_stops_128)


# In[47]:


cTweet_without_stops_128 = sorted(cTweet_without_stops_128, key=lambda kv: kv[1], reverse=True)


# In[48]:


cTweet_without_stops_128[:10] #10 самых популярных после сортировки


# In[49]:


stopwords.words('english').append('is')


# In[50]:


cTweet_without_stops_128[-20:] #20 самых непопулярных после сортировки


# Сравнивая эти топы с топами, полученными при использовании токенизации по пробелам, можно сказать, что топы, полученные с помощью TweetTokenizer с фильтрацией пунктуации и фильтрацией по количеству символов, лучше подходят для анализа твитов, так как наполнены релевантнами периоду пандемии словами. С другой стороны, если например стоит задача анализа употребления людьми предлогов или пунктуационных знаков, то стоит использовать самую простую токенизацию и не фильтровать словарь слов.

# Выведите топ-10 популярных хештегов с количеством встреч. Что можно сказать о них?

# In[51]:


#выведем топ хештэгов:
hashtags = []
for word_7 in cTweet_without_stops_128:
    if word_7[0].startswith("#"):
        hashtags.append(word_7)


# In[52]:


hashtags = sorted(hashtags, key=lambda kv: kv[1], reverse=True)


# In[53]:


hashtags[:10]


# 9 из 10 хэштегов связаны напрямую с упоминанием COVID. Это логично, так как в датасете собраны твиты о коронавирусе.

# То же самое проделайте для ссылок на сайт https://t.co Сравнима ли популярность ссылок с популярностью хештегов? Будет ли информация о ссылке на конкретную страницу полезна?

# In[54]:


#выведем топ ссылок на сайт:
links = []
for word_li in cTweet_without_stops_128:
    if word_li[0].startswith("https://t.co"):
        links.append(word_li)
links = sorted(links, key=lambda kv: kv[1], reverse=True)
links[:10]


# Популярность ссылок несравнима с популярностью хештэгов. Хештэги популярнее в тысячи раз. Это объясняется тем, что ссылка нужна только когда человек хочет поделиться материалами со стороннего ресурса(это редко в твитах), а хештэг можно добавить к любому посту(причем не обязательно 1). С помощтю хештэгов люди задают тему своего поста, а также помещают пост в подборки по хештэгам.

# Информация о ссылке на конкретную страницу будет полезна, если например по текту твита не удается определить эмоциональный окрас твита. Перейдя по ссылке можно понять, что хотел сказать автор твита.

# Используем опыт предыдущих экспериментов и напишем собственный токенайзер, улучшив TweetTokenizer. Функция tokenize должна:
# 
# 
# 
# *   Привести текст в нижний регистр
# *   Применить TweetTokenizer для изначального выделения токенов
# *   Удалить стоп-слова, пунктуацию, токены из одного символа, с позицией в таблице Unicode 128 и более и ссылки на t.co

# In[55]:


#пишем свой токенайзер:


# In[56]:


def custom_tokenizer(text):

     #приводим текст в нижний регистр
    text = text.lower()
    
    
     #применяем TweetTokenizer
    twT = TweetTokenizer()
    text_TT = twT.tokenize(text)
  
    
     #чистим текст
    for slovo in text_TT:
         if slovo in stopwords.words('english'):
             text_TT.remove(slovo)    
    for slovo in text_TT:
         if slovo in punctuation:
             text_TT.remove(slovo)
    
    for slovo in text_TT:
         if slovo.isascii() != True or len(slovo) == 1:
             text_TT.remove(slovo)
    for slovo in text_TT:
         if slovo.startswith("https://t.co/"):
             text_TT.remove(slovo)
    for slovo in text_TT:
         if slovo == '\x92':
             text_TT.remove(slovo)        
            
        
                       
        
    

    return text_TT


# In[57]:


custom_tokenizer('This is sample text!!!! @Sample_text I, \x92\x92 https://t.co/sample  #sampletext')


# In[58]:


#не понимаю, почему is не убрался, хотя он есть в списке стоп слов:(


# ## Задание 3 Векторизация текстов (2 балла)

# Обучите CountVectorizer с использованием custom_tokenizer в качестве токенайзера. Как размер полученного словаря соотносится с размером изначального словаря из начала задания 2?

# In[61]:


text1


# In[70]:


from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(tokenizer = custom_tokenizer)
bow = cv.fit(train['OriginalTweet']) # bow — bag of words (мешок слов)
#bow_test = cv.transform(test)

#scaler = MaxAbsScaler()
#bow = scaler.fit_transform(bow)
#bow_test = scaler.transform(bow_test)

print(len(cv.vocabulary_))


# Посмотрим на какой-нибудь конкретный твитт:

# In[60]:


ind = 9023
train.iloc[ind]['OriginalTweet'], train.iloc[ind]['Emotion']


# Автор твитта не доволен ситуацией с едой во Франции и текст имеет резко негативную окраску.
# 
# Примените обученный CountVectorizer для векторизации данного текста, и попытайтесь определить самый важный токен и самый неважный токен (токен, компонента которого в векторе максимальна/минимальна, без учета 0). Хорошо ли они определились, почему?

# In[71]:


#fr_str = str(train.iloc[ind]['OriginalTweet'])


# In[72]:


#fr_str


# In[73]:


#fr = fr_str.split()


# In[74]:


#france = bow.fit_transform(fr)


# In[78]:


from sklearn.preprocessing import MaxAbsScaler


# In[75]:


#bow_test = cv.transform(fr)

#scaler = MaxAbsScaler()
#bow = scaler.fit_transform(bow)
#bow_test = scaler.transform(fr)


# In[162]:


#france


# In[163]:


#print(france)


# In[65]:


#text1.split('.')


# In[76]:


#train


# In[84]:


train['OriginalTweet'][9023]


# Применим обученный векторайзер для векторизации твита про Францию:

# In[86]:


france_post = train.iloc[ind]['OriginalTweet']


# In[87]:


test_l = []
test_l.append(france_post)


# In[89]:


france_1 = cv.transform(test_l)

#scaler = MaxAbsScaler()
#bow = scaler.fit_transform(bow)
#france_1 = scaler.transform(france_1)


# In[90]:


france_1


# In[91]:


list(cv.vocabulary_.items())[:10]


# In[92]:


print(france_1)


# В этом сообщении каждое слово встречается не более одного раза, поэтому векторайзер выдаёт единицы. Значит, здесь каждого слова из тех, что есть в твите, важность одинаковая.

# Теперь примените TfidfVectorizer и определите самый важный/неважный токены. Хорошо ли определились, почему?

# **Теперь применим Tfidf:**

# In[94]:


from sklearn.feature_extraction.text import TfidfVectorizer


# In[96]:


Tf_Idf = TfidfVectorizer(ngram_range=(1, 1))
tf_train = Tf_Idf.fit(train['OriginalTweet'])


# In[97]:


print(len(Tf_Idf.vocabulary_))


# In[98]:


france_tf = Tf_Idf.transform(test_l)


# In[99]:


france_tf


# In[100]:


print(france_tf)


# Здесь дробные числа,так как количество встреч слова делится на количество слов в твите.

# In[101]:


test_l


# In[103]:


tf_data = pd.DataFrame(france_tf.toarray(), columns=Tf_Idf.get_feature_names_out())


# In[104]:


tf_data.loc[:, (tf_data != 0).all()]


# самым важным токеном по мнению векторайзера оказалось слово brainless. Это разумно и правильно, так как с точки зрения влияния на определение позитивный или негативный окрас имеет твит это слово здесь самое важное.

# самым неважным токеном оказалось слово in. Это тоже имеет смысл, так как предлог может быть в предложении любой тональности и не поможет определить тональность.

# **Найдите какой-нибудь положительно окрашенный твитт, где TfidfVectorizer хорошо (полезно для определения окраски) выделяет важный токен, поясните пример.**
# 
# Подсказка: явно положительные твитты можно искать при помощи положительных слов (good, great, amazing и т. д.)

# **Возьмем такой твит**:

# In[106]:


train


# In[204]:


df_positive = train[train['OriginalTweet'].str.contains("good") | df['OriginalTweet'].str.contains('great') | df['OriginalTweet'].str.contains('amazing')] 
print(df_positive) 


# In[205]:


train['OriginalTweet'][9930]


# In[206]:


print(df_positive.sample(17))


# **Возьмем такой твит:**

# In[116]:


train['OriginalTweet'][9930]


# In[208]:


train['OriginalTweet'][31278]


# он положителен. Протестируем на нем TfIdf:

# In[209]:


posi_post = train['OriginalTweet'][31278]


# In[210]:


test_posi = []
test_posi.append(posi_post)


# In[211]:


test_posi


# In[212]:


posi_tf = Tf_Idf.transform(test_posi)


# In[213]:


posi_tf


# In[214]:


print(posi_tf)


# In[215]:


tf_posi_data = pd.DataFrame(posi_tf.toarray(), columns=Tf_Idf.get_feature_names_out())


# In[216]:


tf_posi_data.loc[:, (tf_posi_data != 0).all()]


# In[217]:


tf_posi_data.iloc[0].idxmax()


# In[241]:


df_good = train[train['OriginalTweet'].apply(lambda x: 'amazing' in x) & (train['Emotion'] == 1)]


# In[242]:


df_good.sample(19)


# In[250]:


good_post = df_good['OriginalTweet'][8234]


# In[251]:


good_post


# In[252]:


test_good = []
test_good.append(good_post)


# In[253]:


good_tf = Tf_Idf.transform(test_good)
tf_good_data = pd.DataFrame(good_tf.toarray(), columns=Tf_Idf.get_feature_names_out())
tf_good_data.loc[:, (tf_good_data != 0).all()]


# In[254]:


tf_good_data.iloc[0].idxmax()


# В этом примере Векторайзер выделяет важные для тональности текста токены большим весом: pensioners связано с помощью пенсионерам, также большой вес придаётся токенам arm, amazing, так как они с большей вероятностью указывают на положительный тон текста 

# ## Задание 4 Обучение первых моделей (1 балл)

# Примените оба векторайзера для получения матриц с признаками текстов. Выделите целевую переменную.

# Получим матрицы с помощью метода get_feature_names_out

# In[261]:


train_tweets = train['OriginalTweet'].tolist()
        
        


# In[262]:


train_tweets


# In[263]:


len(train_tweets)


# In[264]:


tf_for_matrix = Tf_Idf.transform(train_tweets)
Tf_Idf.get_feature_names_out()


# In[265]:


print(tf_for_matrix.shape)


# In[266]:


#то же самое для count vectorizer
cv_for_matrix = cv.transform(train_tweets)
cv.get_feature_names_out()


# In[267]:


print(cv_for_matrix.shape)


# Целевая переменная - это Emotion

# In[270]:


y_train = train['Emotion']


# In[271]:


y_test = test['Emotion']


# Обучите логистическую регрессию на векторах из обоих векторайзеров. Посчитайте долю правильных ответов на обучающих и тестовых данных. Какой векторайзер показал лучший результат? Что можно сказать о моделях?

# In[272]:


#на основе семинарского ноутбука
count_vec = CountVectorizer(tokenizer = custom_tokenizer)
bow_cv = count_vec.fit_transform(train['OriginalTweet']) # bow — bag of words (мешок слов)
bow_cv_test = count_vec.transform(test['OriginalTweet'])

scaler = MaxAbsScaler()
bow_cv = scaler.fit_transform(bow_cv)
bow_cv_test = scaler.transform(bow_cv_test)


# In[273]:


list(count_vec.vocabulary_.items())[:10]


# In[274]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[277]:


from sklearn.metrics import classification_report, roc_auc_score


# **Обучаем логистическую регрессию для Count Vectorizer**

# In[278]:


clf_cv = LogisticRegression(max_iter=200, random_state=42)
clf_cv.fit(bow_cv, y_train)
pred = clf_cv.predict(bow_cv_test)
print(classification_report(y_test, pred))


# Метрики качества модели на тестовых данных говорят о хорошем качестве модели. f1, precision, recall >0.8, значит модель обучилась хорошо

# **Обучаем логистическую регрессию для Tfidf Vectorizer**

# In[279]:


Tf_vec = TfidfVectorizer(ngram_range=(1, 1), tokenizer = custom_tokenizer)
Tf_train = Tf_vec.fit_transform(train['OriginalTweet'])
Tf_test = Tf_vec.transform(test['OriginalTweet'])


scaler = MaxAbsScaler()
Tf_train = scaler.fit_transform(Tf_train)
Tf_test = scaler.transform(Tf_test)


# In[280]:


clf_Tf = LogisticRegression(max_iter=300, random_state=42)
clf_Tf.fit(Tf_train, y_train)
pred_tfidf = clf_Tf.predict(Tf_test)
print(classification_report(y_test, pred_tfidf))


# Метрики качества у лог регрессий, обученных на основе двух разных векторайзеров, почти одинаковые. 

# # Задание 5 Стэмминг 

# Для уменьшения словаря можно использовать стемминг.
# 
# Модифицируйте написанный токенайзер, добавив в него стемминг с использованием SnowballStemmer. Обучите Count- и Tfidf- векторайзеры. Как изменился размер словаря?

# In[281]:


from nltk.stem.snowball import SnowballStemmer


# In[ ]:


def custom_stem_tokenizer(text):
  # -- YOUR CODE HERE --
  return tokens


# In[287]:


def custom_stem_tokenizer(text):

     #приводим текст в нижний регистр
    text = text.lower()
    
    
     #применяем TweetTokenizer
    twT = TweetTokenizer()
    tokens = twT.tokenize(text)
  
    
     #чистим текст
    for slovo in tokens:
         if slovo in stopwords.words('english'):
             tokens.remove(slovo)    
    for slovo in tokens:
         if slovo in punctuation:
             tokens.remove(slovo)
            
    #применяем стэмминг:
    stemmer = SnowballStemmer('english')

    tokens = [stemmer.stem(slovo) for slovo in tokens]
        
    
    for slovo in tokens:
         if slovo.isascii() != True or len(slovo) == 1:
             tokens.remove(slovo)
    for slovo in tokens:
         if slovo.startswith("https://t.co/"):
             tokens.remove(slovo)
    for slovo in tokens:
         if slovo == '\x92':
             tokens.remove(slovo)
    for slovo in tokens:
         if slovo == 'is':
             tokens.remove(slovo)        
            
        
                       
        
    

    return tokens


# In[288]:


custom_stem_tokenizer('This is sample text!!!! @Sample_text I, \x92\x92 https://t.co/sample  #sampletext adding more words to check stemming')


# In[289]:


cv_stem = CountVectorizer(tokenizer = custom_stem_tokenizer)
bow_stem = cv_stem.fit(train['OriginalTweet']) # bow — bag of words (мешок слов)
#bow_test = cv.transform(test)

#scaler = MaxAbsScaler()
#bow = scaler.fit_transform(bow)
#bow_test = scaler.transform(bow_test)

print(len(cv_stem.vocabulary_))


# Размер словаря существенно уменьшился

# теперь обучим с новым токенайзером Tfidf векторайзер:

# In[290]:


Tf_Idf_stem = TfidfVectorizer(tokenizer = custom_stem_tokenizer)
tf_train_stem = Tf_Idf_stem.fit(train['OriginalTweet'])
print(len(Tf_Idf_stem.vocabulary_))


# Обучите логистическую регрессию с использованием обоих векторайзеров. Изменилось ли качество? Есть ли смысл применять стемминг?

# Теперь обучим лог регрессии с новым токенайзером:

# In[291]:


#count_vec_stem = CountVectorizer(tokenizer = custom_tokenizer)
bow_cv_stem = cv_stem.fit_transform(train['OriginalTweet']) # bow — bag of words (мешок слов)
bow_cv_test_stem = cv_stem.transform(test['OriginalTweet'])

scaler = MaxAbsScaler()
bow_cv_stem = scaler.fit_transform(bow_cv_stem)
bow_cv_test_stem = scaler.transform(bow_cv_test_stem)


# In[332]:


bow_cv_stem.shape


# In[333]:


bow_cv_test_stem.shape


# In[293]:


#с помощью Count Vectorizer:
clf_cv_stem = LogisticRegression(max_iter=200, random_state=42)
clf_cv_stem.fit(bow_cv_stem, y_train)
pred_cv_stem = clf_cv_stem.predict(bow_cv_test_stem)
print(classification_report(y_test, pred_cv_stem))


# Изменения почти не видны(только +0.01 в f1)

# In[294]:


#Tf_vec = TfidfVectorizer(ngram_range=(1, 1), tokenizer = custom_tokenizer)
Tf_train_stem = Tf_Idf_stem.fit_transform(train['OriginalTweet'])
Tf_test_stem = Tf_Idf_stem.transform(test['OriginalTweet'])


scaler = MaxAbsScaler()
Tf_train_stem = scaler.fit_transform(Tf_train_stem)
Tf_test_stem = scaler.transform(Tf_test_stem)


# In[295]:


#с помощью Tfidf:
clf_tf_stem = LogisticRegression(max_iter=200, random_state=42)
clf_tf_stem.fit(Tf_train_stem, y_train)
pred_tf_stem = clf_tf_stem.predict(Tf_test_stem)
print(classification_report(y_test, pred_tf_stem))


# Изменения не существенны. Значит,нет смысла применять стэмминг.

# # Задание 6 Работа с частотами (1.5 балла)

# In[296]:


# на данный момент размер словаря 37016


# Еще один способ уменьшить количество признаков - это использовать параметры min_df и max_df при построении векторайзера эти параметры помогают ограничить требуемую частоту встречаемости токена в документах.
# 
# По умолчанию берутся все токены, которые встретились хотя бы один раз.
# 
# Подберите max_df такой, что размер словаря будет на 1 меньше, чем было. Почему параметр получился такой большой/маленький?

# In[301]:


cv_df = CountVectorizer(tokenizer=custom_stem_tokenizer,
                        max_df=10000
                        ).fit(
                            train['OriginalTweet']
                            )
print(len(cv_df.vocabulary_))


# Размер словаря уменьшился на 1. Параметр max_df получился маленький, так как все слова,кроме одного,  встречавшиеся больше 10000 раз ушли при фильтрации новым токенайзером.

# **Подберите min_df (используйте дефолтное значение max_df) в CountVectorizer таким образом, чтобы размер словаря был 3700 токенов (при использовании токенайзера со стеммингом), а качество осталось таким же, как и было. Что можно сказать о результатах?**

# In[307]:


cv_df_2 = CountVectorizer(tokenizer=custom_stem_tokenizer,
                        min_df=11
                        ).fit(
                            train['OriginalTweet']
                            )
print(len(cv_df_2.vocabulary_))


# значение 3700 лежит между min_df =11 и =12. Возьмём =11

# **В предыдущих заданиях признаки не скалировались. Отскалируйте данные (при словаре размера 3.7 тысяч, векторизованные CountVectorizer), обучите логистическую регрессию, посмотрите качество и выведите berplot содержащий по 10 токенов, с наибольшим по модулю положительными/отрицательными весами. Что можно сказать об этих токенах?**

# теперь отскалируем данные:

# In[308]:


from sklearn.preprocessing import StandardScaler


# In[312]:


cv_df_2


# In[313]:


cv_df_CV = CountVectorizer(tokenizer=custom_stem_tokenizer, min_df=11)


# In[316]:


CV_fit_transform = cv_df_CV.fit_transform(train['OriginalTweet'])


# In[336]:


#CV_fit_transform


# In[335]:


#CV_fit_transform_test


# In[319]:


s_scaler = StandardScaler(with_mean=False)
ss_fit = s_scaler.fit(CV_fit_transform)
scaled_cv_df_2 = ss_fit.transform(CV_fit_transform)


# In[340]:


#scaled_cv_df_2.iloc[:, :2260]


# In[321]:


print(scaled_cv_df_2)


# In[362]:


#отскалируем тестовую выборку
CV_fit_transform_test = cv_df_CV.transform(test['OriginalTweet'])
ss_test = s_scaler.transform(CV_fit_transform_test)
#scaled_cv_df_2_test = ss_fit_test.transform(CV_fit_transform_test)


# In[357]:


#bow_cv_test_stem = cv_stem.transform(test['OriginalTweet'])

#scaler1 = MaxAbsScaler()
#scaled_cv_df_2 = scaler1.fit_transform(scaled_cv_df_2)
#scaled_cv_df_2_test = scaler1.transform(scaled_cv_df_2_test)


# In[359]:


ss_test.shape


# In[349]:


scaled_cv_df_2


# In[360]:


clf_cv_ss = LogisticRegression(max_iter=200, random_state=42)
clf_cv_ss.fit(scaled_cv_df_2, y_train)
pred_ss = clf_cv_ss.predict(ss_test)
print(classification_report(y_test, pred_ss))


# In[367]:


#test['OriginalTweet']


# In[364]:


#CV_fit_transform_test


# # Задание 7 Другие признаки (1.5 балла)

# In[368]:


df.head()


# Мы были сконцентрированы на работе с текстами твиттов и не использовали другие признаки - имена пользователя, дату и местоположение
# 
# Изучите признаки UserName и ScreenName. полезны ли они? Если полезны, то закодируйте их, добавьте к матрице с отскалированными признаками, обучите логистическую регрессию, замерьте качество.

# In[370]:


df['UserName'].nunique()


# Все пользователи в этом фрейме уникальны, поэтому, как мне кажется, этот признак бесполезен для анализа.

# In[371]:


df['ScreenName'].nunique()


# С признаком Screen Name ситуация такая же.

# Изучите признак TweetAt в обучающей выборке: преобразуйте его к типу datetime и нарисуйте его гистограмму с разделением по цвету на оспнове целевой переменной. Полезен ли он? Если полезен, то закодируйте его, добавьте к матрице с отскалированными признаками, обучите логистическую регрессию, замерьте качество.

# In[372]:


#посмотрим на признак TweetAt:


# In[373]:


train.head()


# In[378]:


train['TweetAt'] = pd.to_datetime(train['TweetAt'], format='mixed')


# In[379]:


train.head()


# In[408]:


from matplotlib.pyplot import figure
figure(figsize=(30, 10), dpi=80)
plt.style.use('bmh')
def plot_beta_hist(ax, a, b):
    ax.hist(x = train[train['Emotion'] == 1]['TweetAt'],
            histtype="stepfilled", bins=30, alpha=0.8, color='green')
    ax.hist(x = train[train['Emotion'] == 0]['TweetAt'],
            histtype="stepfilled", bins=30, alpha=0.8, color='red')


fig, ax = plt.subplots()
plot_beta_hist(ax, 10, 10)
plot_beta_hist(ax, 4, 12)
plot_beta_hist(ax, 50, 12)
plot_beta_hist(ax, 6, 55)
ax.set_title("Количество твитов по месяцам")
plt.legend(["Позитивные твиты", "Негативные твиты"], loc="upper right")
plt.show()


# По графику видно, что позитивных и негативных твитов каждый день примерно равные пропорции. Поэтому этот признак, по моему менению, не будет полезен для анализа в данном случае.

# **Поработайте с признаком Location в обучающей выборке. Сколько уникальных значений?**

# In[409]:


#переходим к анализу признака Location:


# In[411]:


train['Location'].nunique()


# большое количество уникальных значений

# **Постройте гистограмму топ-10 по популярности местоположений (исключая Unknown)**

# In[417]:


popular_locations = (train['Location'].value_counts())[1:11]

popular_locations.plot(kind='bar', color='orange')
plt.ylabel("Количество твитов")
plt.title("Топ-10 популярных локаций авторов твитов")


# In[418]:


#создадим новый столбец с более общими локациями


# Видно, что многие местоположения включают в себя более точное название места, чем другие (Например, у некоторых стоит London, UK; а у некоторых просто UK или United Kingdom).
# 
# Создайте новый признак WiderLocation, который содержит самое широкое местоположение (например, из London, UK должно получиться UK). Сколько уникальных категорий теперь? Постройте аналогичную гистограмму.

# In[423]:


train['WiderLocation'] = train.apply(lambda row: row.Location.split(' ')[-1] if len(row.Location.split(' ')) > 1 else row.Location.split(' ')[0], axis = 1)
 
# Print the DataFrame after addition
# of new column
train


# In[424]:


train.WiderLocation.unique()


# In[425]:


len(train.WiderLocation.unique())


# теперь уникальных категорий намного меньше, это должно улучшить качество анализа

# In[433]:


popular_wider_locations = (train['WiderLocation'].value_counts())[1:11]

popular_wider_locations.plot(kind='bar', color='blue')
plt.ylabel("Количество твитов")
plt.title("Топ-10 популярных локаций авторов твитов")


# In[435]:


from sklearn.preprocessing import OneHotEncoder


# In[447]:


train = train.loc[:, :'Emotion']


# In[448]:


train.head()


# In[449]:


train['WiderLocation'] = train.apply(lambda row: row.Location.split(' ')[-1] if len(row.Location.split(' ')) > 1 else row.Location.split(' ')[0], axis = 1)
 
# Print the DataFrame after addition
# of new column
train


# In[455]:


train.head()


# In[456]:


#здесь я потерял исходный датафрейм train и не понял как его восстановить(((


# In[ ]:




