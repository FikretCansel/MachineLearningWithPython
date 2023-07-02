#basic imports
import glob
import IPython
from random import randint

#data processing
import librosa
import numpy as np

#modelling
from sklearn.model_selection import train_test_split

from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Flatten
from keras.models import Model
from keras.optimizers import RMSprop

IPython.display.Audio("data/audio/Dogs/dog_barking_0.wav")


def audio2vector(file_path, max_pad_len=400):
    # read the audio file
    audio, sr = librosa.load(file_path, mono=True)
    # reduce the shape
    audio = audio[::3]

    # extract the audio embeddings using MFCC
    mfcc = librosa.feature.mfcc(audio, sr=sr)

    # as the audio embeddings length varies for different audio, we keep the maximum length as 400
    # pad them with zeros
    pad_width = max_pad_len - mfcc.shape[1]
    mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
    return mfcc

audio_file = 'data/audio/Dogs/dog_barking_0.wav'

audio2vector(audio_file)


def get_training_data():
    pairs = []
    labels = []

    Dogs = glob.glob('data/audio/Dogs/*.wav')
    Sub_dogs = glob.glob('data/audio/Sub_dogs/*.wav')
    Cats = glob.glob('data/audio/Cats/*.wav')

    np.random.shuffle(Sub_dogs)
    np.random.shuffle(Cats)

    for i in range(min(len(Cats), len(Sub_dogs))):
        # imposite pair
        if (i % 2) == 0:
            pairs.append([audio2vector(Dogs[randint(0, 3)]), audio2vector(Cats[i])])
            labels.append(0)

        # genuine pair
        else:
            pairs.append([audio2vector(Dogs[randint(0, 3)]), audio2vector(Sub_dogs[i])])
            labels.append(1)

    return np.array(pairs), np.array(labels)



X, Y = get_training_data()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def build_base_network(input_shape):
    input = Input(shape=input_shape)
    x = Flatten()(input)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.1)(x)
    x = Dense(128, activation='relu')(x)
    return Model(input, x)

input_dim = X_train.shape[2:]

audio_a = Input(shape=input_dim)
audio_b = Input(shape=input_dim)


base_network = build_base_network(input_dim)

feat_vecs_a = base_network(audio_a)
feat_vecs_b = base_network(audio_b)


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))


def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)


distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])


epochs = 13
rms = RMSprop()


model = Model(input=[audio_a, audio_b], output=distance)


def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


model.compile(loss=contrastive_loss, optimizer=rms)

audio_1 = X_train[:, 0]
audio_2 = X_train[:, 1]


model.fit([audio_1, audio_2], y_train, validation_split=.25,
          batch_size=128, verbose=2, nb_epoch=epochs)

