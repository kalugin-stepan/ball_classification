import tensorflow as tf
from tensorflow.python import keras
from tensorflow.keras.layers import Dense, Flatten

model = keras.Sequential([
    Flatten(input_shape=(10, 10, 1)),
    Dense(15, activation='relu'),
    Dense(2, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

train_ds = tf.keras.utils.image_dataset_from_directory(
    'dataset',
    validation_split=0.2,
    seed=123,
    color_mode='grayscale',
    label_mode='categorical',
    subset='training',
    image_size=(10, 10)
)

test_ds = tf.keras.utils.image_dataset_from_directory(
    'test_data',
    validation_split=0.2,
    seed=123,
    color_mode='grayscale',
    label_mode='categorical',
    subset='validation',
    image_size=(10, 10)
)

model.fit(
    train_ds,
    epochs=5
)

print(model.evaluate(test_ds))

model.save('better_model')