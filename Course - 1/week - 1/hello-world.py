import tensorflow as tf
import numpy as np
from tensorflow import keras
import math

x = int(input("Enter a random number : "))
# Defining Neraul Network
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# Defining Data
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# Train the model
model.fit(xs, ys, epochs=1000)

print(math.ceil(model.predict([x])))
