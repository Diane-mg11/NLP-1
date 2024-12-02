

df = pd.read_csv(r'C:\Users\dream\Desktop\projects\0.data\nlp\movie_data.csv', encoding='utf-8')

X = df['review']
y = df['sentiment']
print(X)

import re

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)  
    text = re.sub('[\W]+', ' ', text.lower())  
    return text.strip()  


X = X.map(preprocessor)

def word_tokenize(text):
    return text.split(' ')  # 將文本分割成單詞

# 將所有影評分詞
A = [word_tokenize(X[i]) for i in range(X.shape[0])]

from gensim.models import Word2Vec

#  Word2Vec 
model = Word2Vec(A, vector_size=100, window=5, min_count=1, sg=1)

import numpy as np

def get_vector(sentence):
    words = word_tokenize(sentence)
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

# 將所有影評轉換為向量表示
X_vectors = np.array([get_vector(X[i]) for i in range(len(X))])


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# split
X_train, X_test, y_train, y_test = train_test_split(X_vectors, y, test_size=0.1, random_state=1)


# 訓練模型
# logistic model 
lr = LogisticRegression()
lr.fit(X_train, y_train)

print(lr.score(X_test, y_test)) #accuracy

# random forest 
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score

rfgs = GridSearchCV(estimator=RandomForestClassifier(random_state=1),
                  param_grid=[{'max_depth': [5, 6, 7],
                         'n_estimators': [100, 200]}],
                  scoring='accuracy',
                  cv=5) 

rfgs.fit(X_train, y_train)
print(rfgs.best_score_)
print(rfgs.best_params_)


