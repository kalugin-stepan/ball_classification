import cv2
from tflite_support.task import core
from tflite_support.task import vision
import time

base_options = core.BaseOptions('model.tflite', use_coral=False, num_threads=4)

options = vision.ImageClassifierOptions(base_options)

clasifier = vision.ImageClassifier.create_from_options(options)

img = cv2.imread('ball.jpg')

def detect(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    input_tensor = vision.TensorImage.create_from_array(img)

    rez = clasifier.classify(input_tensor).classifications[0].categories[1]

    if rez.score > 0.4:
        return 'ball'
    else:
        return 'not ball'

start = time.time()
print(detect(img))
print(time.time() - start)