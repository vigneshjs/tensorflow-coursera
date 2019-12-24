import tensorflow as tf
from tensorflow_core.python.keras.api import keras


model = keras.models.Sequential([
    keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300,300,3)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(32, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, (3,3), activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    optimizer = keras.optimizers.RMSprop(lr=0.001),
    loss = "binary_crossentropy",
    metrics = ['accuracy']
)

train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1/255.0)
train_generator = train_datagen.flow_from_directory(
    'Datasets/horse-or-human',
    target_size=(300, 300),
    batch_size=128,
    class_mode="binary"
)

model.fit_generator(
    train_generator,
    steps_per_epoch = 8,
    epochs=1,
    verbose=1
)