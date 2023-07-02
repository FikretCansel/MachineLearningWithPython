from binhex import openrsrc

from keras.datasets import boston_housing

(x_train,y_train),(x_test,y_test) = boston_housing.load_data()

import numpy as np

type(x_train)

print(x_train.shape)

print(y_train.shape)

print(x_test.shape)

print(y_test.shape)

print(x_train[:3,:]) #ilk 3 satırı yazdırır


#Kerasta veriler sayısala dönüştürülmelidir

x_val = x_train[300:,] ## 60 egitim 20 validation 20 test şeklinde 3 parçaya ayrıldı
x_val = x_train[300:,]

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

model.fit(x_train,y_train,batch_size=32,epochs=30) #batch_size => tek seferde tüm verilerin egitilmesi yerine veri setinin alt kümelerine
# egitim yapmak için kullanılan bir tekniktir
#epochs =>
# her bir devirden sonra modeli degerlendirmek için validation verilerini kullanalım

sonuc = model.evaluate(x_test, y_test)

for i in range(len(model.metrics_names)):
    print(model.metrics_names[i],":", sonuc[i])





