import numpy as np
import pandas as pd

df=pd.read_csv('spam.csv',encoding='latin-1')


# print(df.sample(10))
# print(df.shape)
# print(df.info())


# Data Cleaning Part
df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
df.rename(columns={'v1':'target','v2':'text'},inplace=True)
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
df['target']=encoder.fit_transform(df['target'])
# print(df.sample(30))
# print(df.duplicated().sum()) # 403
df=df.drop_duplicates(keep='first')
# print(df.duplicated().sum())

# EDA 
# print(df['target'].value_counts())

import matplotlib.pyplot as plt
# plt.pie(df['target'].value_counts(),labels=['ham','spam'],autopct='%0.2f')

# plt.show()
# data is inbalanced

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')

df['num_characters']=df['text'].apply(len)
# print(df['num_characters'])
# print(df.head())

df['num_words']=df['text'].apply(lambda x: len(nltk.word_tokenize(x)))


df['num_sentences']=df['text'].apply(lambda x: len(nltk.sent_tokenize(x)))

# print(df.info()) 

# print(df[df['target']==0][['num_characters','num_words','num_sentences']].describe()
# )
# print(df[df['target']==1][['num_characters','num_words','num_sentences']].describe()
# )


# Data Preprocessing
# Lower case
# Tokenization
# Removing special characters
# Removing stop words and punctuation
# Stemming

from nltk.corpus import stopwords
import string
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()


def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y[:]
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)
            
    text = y[:]
    y.clear()
    
    for i in text:
        y.append(ps.stem(i))
    
            
    return " ".join(y)

df['transformed_text']=df['text'].apply(transform_text)




# spam_corpus = []
# for msg in df[df['target'] == 1]['transformed_text'].tolist():
#     for word in msg.split():
#         spam_corpus.append(word)
        
# # print(len(spam_corpus))
# import seaborn as sns
# from collections import Counter
# sns.barplot(pd.DataFrame(Counter(spam_corpus).most_common(30))[0],pd.DataFrame(Counter(spam_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()


# ham_corpus = []
# for msg in df[df['target'] == 0]['transformed_text'].tolist():
#     for word in msg.split():
#         ham_corpus.append(word)
        

# from collections import Counter
# sns.barplot(pd.DataFrame(Counter(ham_corpus).most_common(30))[0],pd.DataFrame(Counter(ham_corpus).most_common(30))[1])
# plt.xticks(rotation='vertical')
# plt.show()


#  Model Building
