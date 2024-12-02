import pandas as pd
import numpy as np
df = pd.read_csv(r'C:\Users\dream\Desktop\projects\0.data\nlp\movie_data.csv', encoding='utf-8')

import re
re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', 'aaaa;-Dbbbccc:-);-(ttt=-):-D:P:-(xxxyyyzzz')

def preprocessor(text):
    text = re.sub('<[^>]*>', '', text) # re.sub 取代 <>
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text) # :-) ;-( =-D :-P :D :-(
    text = (re.sub('[\W]+', ' ', text.lower()) +     # [\W]+: 1或多個非數字、字母、底線
            ' '.join(emoticons).replace('-', ''))
    return text

# test
print(preprocessor("</a>This :) is :( a test?Yes :-)!"))


