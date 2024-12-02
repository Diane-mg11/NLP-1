import pandas as pd
import numpy as np


# bag-of-words model
from sklearn.feature_extraction.text import CountVectorizer

count = CountVectorizer()
docs = np.array([
        'The sun is shining',
        'The weather is sweet',
        'The sun is shining, the weather is sweet, and one and one is two'])
bag = count.fit_transform(docs) # tf
print(bag.shape)

print(sorted(count.vocabulary_.items(), key=lambda x:x[1]))

print(bag)
print("==")
print(bag.toarray())


# Assessing word relevancy via tfidf
np.set_printoptions(precision=2)
from sklearn.feature_extraction.text import TfidfTransformer # tfidf

tfidf = TfidfTransformer(use_idf=True,
                         norm='l2', # L2標準化，向量平方和=1 # 比較不同長度向量
                         smooth_idf=True)
print(tfidf.fit_transform(bag).toarray()) 

# 分開做
tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True) # 保留原始數據完整性
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()
l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2)) # L2標準化


