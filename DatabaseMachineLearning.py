from MssqlConnection import getProductDemo

veri=getProductDemo()

from pandas import read_sql, DataFrame

# print(veri.dtypes)

print(veri.dtypes)

# degiskenler=["FanSpeed","MaterialSpeed","Heat"]
#
# #data split
# from sklearn.model_selection import train_test_split
# train_data,test_data=train_test_split(veri,train_size=0.9,test_size=0.1,random_state=0)
#
#
# from sklearn.linear_model import LinearRegression
# my_model = LinearRegression()
# def fit_data():
#     my_model.fit(train_data[degiskenler], train_data["Quality"])
#
# fit_data()
#
#
# def predict_quality(fanSpeed,materialSpeed,heat):
#     d = {'FanSpeed': [fanSpeed], 'MaterialSpeed': [materialSpeed],'Heat':[heat]}
#     predict_dataframe=DataFrame(data=d)
#     print("######PredictData########")
#     print(predict_dataframe)
#     print("Tahmin edilen deger :")
#     predictedData=my_model.predict(predict_dataframe)[0]
#     print(predictedData)
#     return predictedData
#
# predict_quality(100,50,65)
# #R KARE TEST
#
# print('R Kare (eğitim) {:.3f}'   .format(my_model.score(train_data[degiskenler],train_data["Quality"])))
# print('R Kare (test) {:.3f}'   .format(my_model.score(test_data[degiskenler],test_data["Quality"])))
