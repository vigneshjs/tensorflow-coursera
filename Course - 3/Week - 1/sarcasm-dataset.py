import os
import json

import numpy as np
from tensorflow_core.python.keras.api import keras
from tensorflow_core.python.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow_core.python.keras.api.keras.preprocessing.sequence import pad_sequences

base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
train_data = os.path.join(base_dir, 'Datasets/news-headlines-dataset-for-sarcasm-detection/Sarcasm_Headlines_Dataset_v2.json')

datastore=[]
headline=[]
is_sarcastic=[]
article_link=[]

training_size=20000
padding_type='post'
truncating_type='post'
max_length=32

vocab_size=1000
embedding_dimension=16

with open(train_data, 'r') as f:
    for line in f:
        datastore.append(json.loads(line))

for data in datastore:
    headline.append(data['headline'])
    is_sarcastic.append(data['is_sarcastic'])
    article_link.append(data['article_link'])


training_sentences=headline[0:training_size]
training_labels=np.array(is_sarcastic[0:training_size])
testing_sentences=headline[training_size:]
testing_labels=np.array(is_sarcastic[training_size:])

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<oov>')
tokenizer.fit_on_texts(training_sentences)

training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = pad_sequences(
    training_sequences,
    padding=padding_type,
    maxlen=max_length,
    truncating=truncating_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(
    testing_sequences,
    padding=padding_type,
    maxlen=max_length,
    truncating=truncating_type
)

model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dimension, input_length=max_length),
    keras.layers.GlobalAveragePooling1D(),
    keras.layers.Dense(24, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.fit(
    training_padded,
    training_labels,
    epochs=10,
    validation_data=(testing_padded, testing_labels),
    verbose=1
)

 