import tensorflow as tf
import numpy as np
from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

xs = np.array([0.0, 1.0, 3.0, 10.0, 2.0, 12.0, 4.0, 6.0], dtype=float)
ys = np.array([0.5, 1.0, 2.0, 5.5, 1.5, 6.5, 2.5, 3.5], dtype=float)

model.compile(optimizer='sgd', loss='mean_squared_error')

model.fit(xs, ys, epochs=1200)

print(model.predict([5.0]))
