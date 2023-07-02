from pandas import read_csv, DataFrame
import matplotlib.pyplot as plt
from scipy import stats
from MssqlConnection import getRealProducts

veri= getRealProducts()

x = veri["heat"]
y = veri["quality"]

#LinearRegresyon grafik
slope, intercept, r, p, std_err = stats.linregress(x, y)
def myfunc(x):
 return slope * x + intercept

mymodel = list(map(myfunc, x))
plt.scatter(x, y)
plt.plot(x, mymodel)
plt.show()


#Polynomial Regression
import numpy
mymodel = numpy.poly1d(numpy.polyfit(x, y, 3))
myline = numpy.linspace(1, 120, 10)
plt.scatter(x, y)
plt.plot(myline, mymodel(myline))
plt.show()



degiskenler=["fanspeed","materialspeed","heat"]


#Tanımlayıcı istatislikler
from pandas import set_option
set_option("display.width", 100)
description = veri[degiskenler].describe()

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
def fit_data():
    my_model.fit(train_data[degiskenler], train_data["quality"])

fit_data()


def predict_quality(fanSpeed,materialSpeed,heat):
    d = {'fanspeed': [fanSpeed], 'materialspeed': [materialSpeed],'heat':[heat]}
    predict_dataframe=DataFrame(data=d)
    print("######PredictData########")
    print(predict_dataframe)
    print("Tahmin edilen deger :")
    predictedData=my_model.predict(predict_dataframe)[0]
    print(predictedData)
    return predictedData

# predict_quality(100,50,65)
#R KARE TEST

print('R Kare (eğitim) {:.3f}'   .format(my_model.score(train_data[degiskenler],train_data["quality"])))
print('R Kare (test) {:.3f}'   .format(my_model.score(test_data[degiskenler],test_data["quality"])))








