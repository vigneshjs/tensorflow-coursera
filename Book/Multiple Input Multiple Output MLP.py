from tensorflow_core.python.keras.api import keras
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()

train_data, test_data, train_labels, test_labels = train_test_split(
    housing.data, housing.target
)

train_data1, train_data2 = train_data[:, :5], train_data[:, 2:]

input1 = keras.layers.Input(shape=train_data1[1].shape)
input2 = keras.layers.Input(shape=train_data2[1].shape)
hidden1 = keras.layers.Dense(300, activation='relu')(input2)
hidden2 = keras.layers.Dense(30, activation='relu')(hidden1)
concat = keras.layers.Concatenate()([input1, hidden2])
output1 = keras.layers.Dense(1)(concat)
output2 = keras.layers.Dense(1)(hidden2)

model = keras.Model(
    inputs=[input1, input2],
    outputs=[output1, output2]
)

model.compile(
    loss=['mse', 'mse'],
    loss_weights=[0.9, 0.1],
    optimzer="sgd"
)

# keras.utils.plot_model(model)
