import pandas as pd
import numpy as np

# NLTK中提供了三種最常用的詞幹提取器介面，Porterstemmer,LancasterStemmer和SnowballStemmer。

from nltk.stem.porter import PorterStemmer # 基於字尾剝離的詞幹提取

porter = PorterStemmer() 

def tokenizer(text): # 比較有無stemmer的差異
    return text.split() 


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()] 


# compare
print(tokenizer('runners like running and thus they run a lot'))
print("==")
print(tokenizer_porter('runners like running and thus they run a lot'))

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
stop = stopwords.words('english')
# test
[w for w in tokenizer_porter('a runner likes running and runs a lot') if w not in stop]