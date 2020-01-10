import pandas as pd
import numpy as np
import pickle
import re, os
import random
from tensorflow_core.python.keras.api import keras
from tensorflow_core.python.keras.api.keras.preprocessing.text import Tokenizer
from tensorflow_core.python.keras.api.keras.preprocessing.sequence import pad_sequences

# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

DATASET_COLUMNS = ["rating", "ids", "date", "flag", "user", "text"]
TEXT_CLEANING_RE = "@\S+|https?:\S+|http?:\S|[^A-Za-z0-9]+"
stop_words = ["a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did", "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have", "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its", "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's", "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves", "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those", "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've", "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom", "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself", "yourselves" ]

vocab_size = 10000
max_length = 100
embedding_dim = 300
epochs = 10

dataframe = pd.read_csv('/home/vignesh/Downloads/sentiment140/training.1600000.processed.noemoticon.csv', encoding = "ISO-8859-1", names=DATASET_COLUMNS)
datas = dataframe[['rating', 'text']]

def preprocessing(text):
    """
    Remove the username, stop words from the text
    """
    text = re.sub(TEXT_CLEANING_RE, " ", str(text).lower()).strip()
    words = []
    for word in text.split():
        if word not in stop_words:
            words.append(word)

    return " ".join(words)

def preprocessrating(rating):
    if rating > 0:
        return 1
    else:
        return 0

datas.text = datas.text.apply(lambda x: preprocessing(x))
datas.rating = datas.rating.apply(lambda x:preprocessrating(x))

datas = datas.sample(frac=1)

train_data = datas
train_data_rating = np.array(train_data['rating'])
train_data_text = train_data['text']

print(train_data_rating[100], train_data_text[100] )

# test_data = datas[1500000:]
# test_data_rating = np.array(test_data['rating'])
# test_data_text = test_data['text']

tokenizer = Tokenizer(num_words=vocab_size, oov_token="<oov>")
tokenizer.fit_on_texts(train_data_text)

temp_sequence = tokenizer.texts_to_sequences(train_data_text)
train_data_text = pad_sequences(temp_sequence, maxlen=max_length, padding='post')

# temp_sequence_test = tokenizer.texts_to_sequences(test_data_text)
# test_data_text = pad_sequences(temp_sequence_test, maxlen=max_length, padding='post')

model = keras.models.Sequential([
    keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    keras.layers.Dropout(0.2),
    keras.layers.Conv1D(64, 5, activation='relu'),
    keras.layers.MaxPooling1D(pool_size=4),
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss="binary_crossentropy",
    metrics=['accuracy']
)

model.summary()

model.fit(
    train_data_text, train_data_rating, batch_size=128,epochs=epochs
)

model.save('twitter.h5')

with open('twitter-tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

print("ts", train_data_text[0])

