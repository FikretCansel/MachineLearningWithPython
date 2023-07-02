from keras.datasets import boston_housing
(x_train,y_train),(x_test,y_test) = boston_housing.load_data()

import numpy as np


type(x_train)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

#60 egitim , 20 validation , 20 test
x_val = x_train[300:,] ## 60 egitim 20 validation 20 test şeklinde 3 parçaya ayrıldı
y_val = x_train[300:,]

#Sıralı katmanlardan oluşan sinir ağı modeli için :
from keras.models import Sequential

model = Sequential()

from keras.layers import Dense, Activation

model.add(Dense(13, input_dim=13,kernel_initializer="normal",activation="relu"))
model.add(Dense(6,kernel_initializer="normal",activation="relu"))

model.add(Dense(1,kernel_initializer="normal"))
#Ortalama kareler hatası
#iyileştirme algoritmaları
#Adam
#Gradyan azaltma
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mean_absolute_percentage_error"])
#batch => tek seferde eğitilmesi yerine alt kümelerin üzerine egitim yapmak için kullanılan teknik
model.fit(x_train,y_train,batch_size=32,epochs=30)

sonuc = model.evaluate(x_test, y_test)



#https://colab.research.google.com/drive/18WiSw1K0BW3jOKO56vxn11Fo9IyOuRjh#scrollTo=czWvnIANYSqI


for i in range(len(model.metrics_names)):
    print(model.metrics_names[i],":", sonuc[i])



