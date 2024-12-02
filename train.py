# Training models for document classification
import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\dream\Desktop\projects\0.data\nlp\movie_data.csv', encoding='utf-8')

import re
def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # re.sub 取代
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) # :-) ;-( =-D :-P :D :-(
    text = (re.sub('[\W]+', ' ', text.lower()) +
            ' '.join(emoticons).replace('-', ''))
    return text

df['review'] = df['review'].map(preprocessor) 

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')


from nltk.stem.porter import PorterStemmer

porter = PorterStemmer()
def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


from sklearn.model_selection import train_test_split

df_train, df_test = train_test_split(df, test_size=0.1, random_state=1, stratify=df['sentiment'])
print("TRAIN size:", len(df_train))
print("TEST size:", len(df_test))

X_train = df_train['review'].values # values 轉 array
y_train = df_train['sentiment'].values
X_test = df_test['review'].values
y_test = df_test['sentiment'].values

# test 
print(X_train.shape)
print(y_train.shape)


from sklearn.feature_extraction.text import TfidfVectorizer 

tfidf = TfidfVectorizer(stop_words=stop, tokenizer=tokenizer_porter)
X_train = tfidf.fit_transform(X_train).toarray() 

X_test = tfidf.fit_transform(X_test).toarray() 