import matplotlib.pyplot as plt
import os
import tensorflow as tf
from tensorflow_core.python.keras.api.keras import layers
from tensorflow_core.python.keras.api import keras
from tensorflow_core.python.keras.api.keras import Model
from tensorflow_core.python.keras.api.keras.applications.inception_v3 import InceptionV3


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
local_weights_file = os.path.join(BASE_DIR ,'Pre-Trained-Models/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

pre_trained_model = InceptionV3(
    input_shape=(150, 150, 3),
    include_top=False,
    weights=None
)

pre_trained_model.load_weights(local_weights_file)

for layer in pre_trained_model.layers:
    layer.trainable = False

# pre_trained_model.summary()

last_layer = pre_trained_model.get_layer('mixed7')
last_output = last_layer.output

x = layers.Flatten()(last_output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = Model(pre_trained_model.input, x)
model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.RMSprop(lr=0.0001),
    metrics=['acc']
)

base_dir = os.path.join(BASE_DIR ,'Datasets')
training_dir = os.path.join(base_dir, 'horse-or-human')
validation_dir = os.path.join(base_dir, 'validation-horse-or-human')

training_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

training_generator = training_datagen.flow_from_directory(
    training_dir,
    batch_size=5,
    target_size=(150, 150),
    class_mode='binary'
)

validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=4,
    class_mode='binary'
)

history = model.fit_generator(
    training_generator,
    validation_data=validation_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_steps=32,
    verbose=1)

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend(loc=0)
plt.figure()


plt.show()
