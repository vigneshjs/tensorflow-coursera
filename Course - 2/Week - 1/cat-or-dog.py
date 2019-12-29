import tensorflow as tf
from tensorflow_core.python.keras.api import keras
import os


base_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Datasets/cats_and_dogs_filtered')
train_data_path = os.path.join(base_dir, 'train')
validation_data_path = os.path.join(base_dir, 'validation')

training_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255)
validation_datagen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1/255)

training_generator = training_datagen.flow_from_directory(
    train_data_path,
    batch_size=20,
    target_size=(150, 150),
    class_mode='binary'
)
validation_generator = validation_datagen.flow_from_directory(
    validation_data_path,
    batch_size=50,
    target_size=(150, 150),
    class_mode='binary'
)

model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3, 3), activation='relu',
                        input_shape=(150, 150, 3)),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(2, 2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


model.summary()

model.compile(
    loss='binary_crossentropy',
    optimizer=keras.optimizers.RMSprop(lr=0.001),
    metrics=['accuracy']
)

model.fit_generator(
    training_generator,
    steps_per_epoch=100,
    epochs=1,
    validation_data=validation_generator,
    validation_steps=10,
    verbose=2
)
