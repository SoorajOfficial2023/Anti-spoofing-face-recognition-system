import os
import random
import shutil
from itertools import islice

outputFolderPath = 'Dataset/splitData'
inputFolderPath = 'Dataset/all'
splitRatio = {"train":0.7,"val":0.2,"test":0.1}    #train:70%,test:10%,val:30% --- data % to test,train and validation
classes = ["fake","real"]

try:
    shutil.rmtree(outputFolderPath)
    print('Removed directory')
except OSError as e:
    os.mkdir(outputFolderPath)
    
#------------Directiories to create--------
os.makedirs(f"{outputFolderPath}/train/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/train/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/val/labels",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/images",exist_ok=True)
os.makedirs(f"{outputFolderPath}/test/labels",exist_ok=True)

#-----------Get the names------------
listNames = os.listdir(inputFolderPath)

uniqueNames = []
for name in listNames:
    uniqueNames.append(name.split('.')[0])
uniqueNames = list(set(uniqueNames))


#-----------shuffle---------------
random.shuffle(uniqueNames)


#-----------find the number of images for each folder----------
lenData = len(uniqueNames)
lenTrain = int(lenData*splitRatio['train'])
lenVal = int(lenData*splitRatio['val'])
lenTest = int(lenData*splitRatio['test'])
#-----------Put remaining images to training---------------
if lenData != lenTrain+lenVal+lenTest:
    remainig = lenData(lenTrain+lenVal+lenTest)
    lenTrain += remainig

#-----------split the list---------------
lengthToSplit = [lenTrain,lenVal,lenTest]
Input = iter(uniqueNames)
outPut = [list(islice(Input,elem))for elem in lengthToSplit]
print(f"Total images:{lenData} \n split: {len(outPut[0])} {len(outPut[1])} {len(outPut[2])}")
#-----------copy files---------------
sequence = ['train','val','test']
for i,out in enumerate(outPut):
    for fileName in out:
        shutil.copy(f"{inputFolderPath}/{fileName}.jpg",f"{outputFolderPath}/{sequence[i]}/images/{fileName}.jpg")
        shutil.copy(f"{inputFolderPath}/{fileName}.txt",f"{outputFolderPath}/{sequence[i]}/labels/{fileName}.txt")
print('Split process completed')

#-----------creating data.yaml file----------

dataYaml = f'''path: ../Data\n\
train: ../train/images\n\
val: ../val/images\n\
test: ../test/images\n\
\n\
nc : {len(classes)}\n\ 
names : {classes}'''

f = open(f"{outputFolderPath}/data.yaml",'a')
f.write(dataYaml)
f.close()

print(f"Data.yaml fie created")