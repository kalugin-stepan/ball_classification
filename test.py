import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
import time

model = keras.models.load_model('better_model')

imgs_path = 'test_data/not_ball'

imgs = os.listdir(imgs_path)

S = 0

for i in imgs:
    img = cv2.imread(os.path.join(imgs_path, i), cv2.IMREAD_GRAYSCALE)

    start = time.time()

    img = tf.expand_dims(img, 0)

    rez = np.argmax(model.predict(img)[0])

    S += time.time() - start

    if rez == 0:
        print('ball')
        print(i)
        continue
    print('not ball')

S /= len(imgs)

print(S)