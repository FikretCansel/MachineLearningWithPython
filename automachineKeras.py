from MssqlConnection import getRealProducts
from pandas import read_csv

data= getRealProducts()
degiskenler=["fanspeed","materialspeed","heat"]

import numpy as np

from sklearn.model_selection import train_test_split
train_data,test_data=train_test_split(data,train_size=0.9,test_size=0.1,random_state=0)
print(train_data[degiskenler])

print(train_data[["quality"]])




from keras.models import Sequential

model=Sequential() # Tabakalar tek tek eklenerek sıralı tabakalar oluşur

from keras.layers import Dense, Activation    #Bir tabaka modellerin bir araya gelmesinden oluşur
#Dense layerdaki her nöronu bir önceki layera baglayan derin sinir aglarıdır

model.add(Dense(13, input_dim=13,kernel_initializer="normal",activation="relu"))  #Modele tabaka ekleme
#activasyon nörona gelen girdi sinyalini çıktı sinyaline çevirir

model.add(Dense(6,kernel_initializer="normal",activation="relu"))

model.add(Dense(1,kernel_initializer="normal"))

#Optimizasyon fonksiyonu
# Kayıp fonksiyonu
# Performansı belirlemek için Metricdir
model.compile(loss="mean_squared_error",optimizer="adam",metrics=["mean_absolute_percentage_error"])#loss => kayıp fonksiyonu => gerçek çıktı ile egitilen çıktı arasındaki farkı ölçer
#Optimizasyon algoritması
#gradyan azaltma => her egitim örneginden sonra kalite hesaplanır türev kullanılarak gradyan azaltma kullanılır
#gradyan azaltma => daha az kayıp ile en iyi model oluşturulmaya çalışılır
# adam => her parametre için uyumlu ögrenme modeli hesaplanır

model.fit(train_data[degiskenler],train_data["quality"],batch_size=32,epochs=30) #batch_size => tek seferde tüm verilerin egitilmesi yerine veri setinin alt kümelerine
# egitim yapmak için kullanılan bir tekniktir
#epochs =>
# her bir devirden sonra modeli degerlendirmek için validation verilerini kullanalım

