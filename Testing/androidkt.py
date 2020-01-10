import pandas as pd
import numpy as np
from tensorflow_core.python.keras.api import keras

Sequential = keras.models.Sequential 
data_file_path='/home/vignesh/Downloads/amazon-fine-food-reviews/Reviews.csv'
df=pd.read_csv(data_file_path)
df=df.astype(str)
df=df[['Summary','New_Score']]

df.drop_duplicates(subset=['New_Score','Summary'],keep='first',inplace=True) 

df=df.sample(frac=1).reset_index(drop=True)

review=np.array(df['Summary'])
rating=np.array(df['New_Score'].astype(int))

num_of_words=80000
max_len=250

tokenizer = keras.preprocessing.text.Tokenizer(num_words=num_of_words,filters='!"#$%&amp;()*+,-./:;&lt;=>?@[\]^_`{|}~', oov_token='<oov>')

tokenizer.fit_on_texts(review)
review_seq = tokenizer.texts_to_sequences(review)

review_seq_pad = keras.preprocessing.sequence.pad_sequences(review_seq, maxlen=max_len, padding='pre')


# train_x,test_x,train_y,test_y = train_test_split(review_seq_pad, rating_1, test_size=0.20, random_state=42)

epochs = 1
emb_dim = 128
batch_size = 256

model = Sequential([
    keras.layers.Embedding(num_of_words, emb_dim, input_length=review_seq_pad.shape[1]),
    keras.layers.SpatialDropout1D(0.2),
    keras.layers.Bidirectional(keras.layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
print(model.summary())

history = model.fit(review_seq_pad, rating, epochs=epochs, batch_size=batch_size,validation_split=0.2)
model.save('s-Food_Reviews.h5')
import pickle

with open('s-new-tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)