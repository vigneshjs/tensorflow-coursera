import tensorflow_datasets as tfds
from tensorflow_core.python.keras.api import keras

imdb, info = tfds.load('imdb_reviews/subwords8k', with_info=True, as_supervised=True)
tokenizer = info.features['text'].encoder
