import csv
import random


fanSpeed = random.randint(55, 100)
print(fanSpeed)

materialSpeed = random.randint(10, 100)
print(materialSpeed)

heat = random.randint(40, 150)
print(heat)

quality = random.randint(10, 100)
print(heat)




# open the file in the write mode
f = open('datasets/automachineDatas', 'w')

# create the csv writer
writer = csv.writer(f)

# write a row to the csv file

newRow=""

newArrayRow=[]

id=0

for x in range(1000):
    id=id+1
    fanSpeed = random.randint(10, 100)
    print(fanSpeed)

    materialSpeed = random.randint(10, 100)
    print(materialSpeed)

    quality = random.randint(10, 100)
    print(quality)

    heat = random.randint(10, 100)
    print(heat)
    newArrayRow.append(fanSpeed)
    newArrayRow.append(heat)
    newArrayRow.append(materialSpeed)
    newArrayRow.append(quality)
    newRow= newRow+str(id)+"," + str(fanSpeed)+","+str(heat)+","+str(materialSpeed)+","+str(quality)+"\n"




writer.writerow([newRow])




# close the file
f.close()