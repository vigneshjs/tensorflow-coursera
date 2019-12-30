import os
import json

from tensorflow_core.python.keras.api import keras
from tensorflow_core.python.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow_core.python.keras.api.keras.preprocessing.sequence import pad_sequences

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
train_data = os.path.join(base_dir, 'Datasets/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json')

datastore=[]
headline=[]
is_sarcastic=[]
article_link=[]

with open(train_data, 'r') as f:
    for line in f:
        datastore.append(json.loads(line))

for data in datastore:
    headline.append(data['headline'])
    is_sarcastic.append(data['is_sarcastic'])
    article_link.append(data['article_link'])

tokenizer = Tokenizer(oov_token='<oov>')
tokenizer.fit_on_texts(headline)
sequences = tokenizer.texts_to_sequences(headline)
padded = pad_sequences(
    sequences,
    padding='post'
)

print(padded.shape)