# Step 1
import cv2  # working with, mainly resizing, images
import numpy as np  # dealing with arrays
import os  # dealing with directories
from random import shuffle  # mixing up or currently ordered data that might lead our network astray in training.
from tqdm import tqdm  # a nice pretty percentage bar for tasks. Thanks to viewer Daniel BA1/4hler for this suggestion
import tensorflow as tf  # Import Tensorflow
import glob  # This will extract all files from the folder
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
import h5py
from keras.models import model_from_json
from keras.models import load_model
import numpy as np
from keras.preprocessing import image
from keras import backend as K
from keras.utils import img_to_array, load_img

# Make labels specific folders inside the training folder and validation folder. For Example: If you have 0-9 images, then you should make
# 10 folders and name them as 0, 1, 2.......9. Name of these subfolders should be same as name of your labels.
# Make a seperate test folder which will have all random images.

# Step 2
# Load images from folder train folder
counter = 0
for imgtrain in glob.glob(
        "C:/ministdata/*.png"):  # You can check number of data in each labelled folder. Here we are
    cv_imgtrain = cv2.imread(imgtrain)  # doing it for '2' label
    counter += 1
print("total images in the folder = ", counter)
# Calculate shape of train
cv_imgtrain.shape  # shape of kaggle MNIST data base is 28,28,3

# Step 3
# Load images from folder test folder
counter = 0
for imgtest in glob.glob("C:/ministdata/*.png"):
    cv_imgtest = cv2.imread(imgtest)
    counter += 1
print("total images in the folder = ", counter)
# Calculate shape of train
cv_imgtest.shape  # shape of kaggle MNIST data base is 28,28,3

# Step 4
# define dimensions of our input images.
img_width, img_height = 28, 28  # Here this is 28 ,28 because the shape of image is 28,28,3. You can input any shape greater than 28.
# You can give shape 150, 150. It will just take longer time for model to run.

# Step 5
# Define directory
train_data_dir = 'D:/Kaggle_data/MNIST/Train'
validation_data_dir = 'D:/Kaggle_data/MNIST/Validation'
nb_train_samples = 49  # Get this in step 2. These are examples per subfolder inside train data folder
nb_validation_samples = 11  # Get this in step 2. These are examples per subfolder inside validation data folder
epochs = 400  # Define number of epochs
batch_size = 10  # Define batch size. This should be less than the total number of examples in validation and training
# set

# Step 6
# Define channels
if K.image_data_format() == 'channels_first':  # We usually use channel first approach
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Step 7
# Define model from scratch
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(10))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
# As I am using 28,28 input shape. I cannot add more Convo layer to this model. If you take 150,150 shape then you can add more layers.
# ((N+2P-F)/S) + 1 is the formula used. P=0, S=1.


# Step 8
# Generate images from directory
train_datagen = ImageDataGenerator(
    rescale=1. / 255)  # We can augment the number of images with ImageDataGenerator #Google to know more
validation_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('MNIST.h5')

# Step 9
# Test on the image form test folders
test_model = load_model('MNIST.h5')
img = load_img('C:/ministdata/img.png', False, target_size=(img_width, img_height))
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
preds = test_model.predict_classes(x)
prob = test_model.predict_proba(x)
print(preds, prob)