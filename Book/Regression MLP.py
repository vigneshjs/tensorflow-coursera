from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow_core.python.keras.api import keras


housing = fetch_california_housing()

train_data, test_data, train_label, test_label = train_test_split(
    housing.data, housing.target
)

input_ = keras.layers.Input(shape=train_data[1].shape)
hidden1 = keras.layers.Dense(300, activation='relu')(input_)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input_, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.Model(inputs=[input_], outputs=[output])

model.compile(
    loss='rms',
    optimzer="sgd"
)

history = model.fit(
    train_data, train_label, validation_split=0.1
)


