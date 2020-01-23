from tensorflow_core.python.keras.api import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

train_data, test_data, train_labels, test_labels = train_test_split(
    housing.data, housing.target
)

train_data1, train_data2 = train_data[:, :5], train_data[:, 2:]

input_1 = keras.layers.Input(shape=train_data1[1].shape, name="wide")
input_2 = keras.layers.Input(shape=train_data2[1].shape, name="deep")
hidden1 = keras.layers.Dense(300, activation='relu', name="hidden1")(input_2)
hidden2 = keras.layers.Dense(30, activation='relu', name="hidden2")(hidden1)
concat = keras.layers.Concatenate()([input_1, hidden2])
output = keras.layers.Dense(1)(concat)

model = keras.Model(
    inputs=[input_1, input_2], 
    outputs=[output]
)

model.compile(
    loss="mean_squared_error",
    optimzier='sgd',
    metrics=['accuracy']
)

model.summary()

model.fit(
    (train_data1, train_data2), 
    train_labels,
    validation_split=0.1
)