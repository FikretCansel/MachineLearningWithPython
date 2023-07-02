from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import keras
from keras.layers import Input,Dense
from keras.optimizers import SGD



import numpy as np
import pandas as pd

veri = pd.read_csv("./breast-cancer_dataset/breast-cancer-wisconsin.data")

veri.replace("?",-9999,inplace='true')

veriyeni = veri.drop(['1000025'],axis=1)

imp = Imputer(missing)