import numpy as np 
import pandas as pd
import os

from tensorflow import keras
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Dropout, TimeDistributed, Flatten, Input
from tensorflow.keras.preprocessing.sequence import pad_sequences


# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

dframe = pd.read_csv('/home/vignesh/Downloads/entity-annotated-corpus/ner.csv', encoding = "ISO-8859-1", error_bad_lines=False)

dataset=dframe.drop(['Unnamed: 0', 'lemma', 'next-lemma', 'next-next-lemma', 'next-next-pos',
       'next-next-shape', 'next-next-word', 'next-pos', 'next-shape',
       'next-word', 'prev-iob', 'prev-lemma', 'prev-pos',
       'prev-prev-iob', 'prev-prev-lemma', 'prev-prev-pos', 'prev-prev-shape',
       'prev-prev-word', 'prev-shape', 'prev-word',"pos"],axis=1)

dataset.isnull().sum()
dataset.dropna(inplace=True)
dataset['tag'].value_counts()
dataset=dataset.drop(['shape'],axis=1)

class SentenceGetter(object):
    
    def __init__(self, dataset):
        self.n_sent = 1
        self.dataset = dataset
        self.empty = False
        agg_func = lambda s: [(w, t) for w,t in zip(s["word"].values.tolist(), s["tag"].values.tolist())]
        self.grouped = self.dataset.groupby("sentence_idx").apply(agg_func)
        self.sentences = [s for s in self.grouped]
    
    def get_next(self):
        try:
            s = self.grouped["Sentence: {}".format(self.n_sent)]
            self.n_sent += 1
            return s
        except:
            return None

getter = SentenceGetter(dataset)
sentences = getter.sentences

sentences = [[s[0].lower() for s in sent] for sent in getter.sentences]
labels = [[s[1] for s in sent] for sent in getter.sentences]
maxlen = max([len(s) for s in sentences])

words = np.array([x.lower() if isinstance(x, str) else x for x in dataset["word"].values])
words = list(set(words))
words.append('unk')
words.append('pad')
n_words = len(words); n_words

tags = list(set(dataset["tag"].values))
n_tags = len(tags)
n_tags
word2idx = {w: i for i, w in enumerate(words)}
tag2idx = {t: i for i, t in enumerate(tags)}

X = [[word2idx.get(w,'27420') for w in s] for s in sentences]
y = [[tag2idx.get(l) for l in lab] for lab in labels]
X = pad_sequences(maxlen=140, sequences=X, padding="post", value=n_words-1)
y = pad_sequences(maxlen=140, sequences=y, padding="post", value=tag2idx["O"])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)
y_train = keras.utils.to_categorical(y_train)


model = Sequential()

model.add(Embedding(n_words, 50))
model.add(Bidirectional(LSTM(140, return_sequences=True)))
model.add(Bidirectional(LSTM(140, return_sequences=True)))
model.add(TimeDistributed(Dense(n_tags, activation="softmax")))
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
model.fit(X_train, y_train, batch_size=32, epochs=5, verbose=1, validation_split=0.2)






