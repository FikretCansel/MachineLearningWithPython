from pandas import read_csv
import numpy as np
import matplotlib.pyplot as plt

veri=read_csv("2016dolaralis.csv")

x = veri["Gun"]
y = veri["Fiyat"]


print(veri.head())

plt.scatter(x,y)
plt.show()

degiskenler=["Gun"]


print(veri[degiskenler].head())

#Tanımlayıcı istatislikler
from pandas import set_option
set_option("display.width", 100)
description = veri[degiskenler].describe()
print(description)

#Korolasyon

import seaborn as sns
import matplotlib.pyplot as plt

j = veri[degiskenler].corr()
f,ax = plt.subplots(figsize=(9,6))

sns.heatmap(j,annot=True,linewidths=.5,ax=ax)

plt.show()

#data split
from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(veri,train_size=0.9,test_size=0.1,random_state=0)

from sklearn.neural_network import MLPRegressor

my_model = MLPRegressor()

print(train_data["Gun"])

my_model.fit(train_data[degiskenler],train_data["Fiyat"])



test1= veri[veri["Gun"] == 250]


print("Tahmin edilen deger :")
print(my_model.predict(test1[degiskenler])[0])

print("Gerçek Deger")
print(test1["Fiyat"].values[0])

#R KARE TEST

print('R Kare (eğitim) {:.3f}'   .format(my_model.score(train_data[degiskenler],train_data["Fiyat"])))
print('R Kare (test) {:.3f}'   .format(my_model.score(test_data[degiskenler],test_data["Fiyat"])))






