import tensorflow as tf
from tensorflow_core.python.keras.api import keras

training_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)

training_generator