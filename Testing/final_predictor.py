import pickle
import os
from tensorflow_core.python.keras.api import keras

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

model = keras.models.load_model('/home/vignesh/Learning/tensorflow-coursera/twitter.h5')
model.summary()

with open('twitter-tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)

input_sentence = [input("Enter a sentence: ").strip()]
print(input_sentence)
sequence = tokenizer.texts_to_sequences(input_sentence)
padded_sequence = keras.preprocessing.sequence.pad_sequences(sequence, maxlen = 100, padding='post')
print(padded_sequence)
ans = model.predict(padded_sequence)
word_index=tokenizer.word_index
print(ans)
if ans[0] > 0.50:
    print("Positive")
else:
    print("Negative")
