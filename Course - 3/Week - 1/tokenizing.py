import os
import json

from tensorflow_core.python.keras.api import keras
from tensorflow_core.python.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow_core.python.keras.api.keras.preprocessing.sequence import pad_sequences

sentences=[
    'I love my dog',
    'I love my cat abd'
]

tokenizer = Tokenizer(num_words=10, oov_token='<oov>')
tokenizer.fit_on_texts(sentences)
word_index=tokenizer.word_index
# print(word_index)
sequences = tokenizer.texts_to_sequences(sentences)
padded = pad_sequences(sequences)
# print(padded)
