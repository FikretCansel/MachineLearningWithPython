# make a prediction for a new image.
from numpy import argmax
from keras.utils import load_img
from keras.utils import img_to_array
from mnist import getMinIstModel
from keras.models import load_model


# load and prepare the image
def load_image(filename):
    # load the image
    img = load_img(filename, grayscale=True, target_size=(28, 28))
    # convert to array
    img = img_to_array(img)
    # reshape into a single sample with 1 channel
    img = img.reshape(1, 28, 28, 1)
    # prepare pixel data
    img = img.astype('float32')
    img = img / 255.0
    return img


# load an image and predict the class
def findNumber():
    # load the image
    img = load_image('./minIstFolder/img_1.png')
    # load model
    model = getMinIstModel()
    # predict the class
    predict_value = model.predict(img)
    digit = argmax(predict_value)
    print(digit)


# entry point, run the example
findNumber()