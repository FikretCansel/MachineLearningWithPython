import random
import time

from AutoMachine2 import predict_quality
from MssqlConnection import addProductDemo


def createValues():
    fanSpeed = random.randint(10, 100)
    print(fanSpeed)

    materialSpeed = random.randint(10, 100)
    print(materialSpeed)

    heat = random.randint(10, 100)
    print(heat)
    return fanSpeed,materialSpeed,heat





for i in range(15):
    print("Üretim Başladı")
    fanSpeed,materialSpeed,heat=createValues()
    predict_quality_value = predict_quality(fanSpeed, materialSpeed, heat)
    if(predict_quality_value>=60):
        print("Değer Kaydedildi. Deger =>"+str(predict_quality_value))
        addProductDemo(heat, materialSpeed, fanSpeed,predict_quality_value)
    else:
        print("Deger Düşük Yeni değer üretiliyor"+str(predict_quality_value))
    time.sleep(1)