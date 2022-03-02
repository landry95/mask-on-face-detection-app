# -*- coding: utf-8 -*-
"""
Created on Tue Mar  1 18:39:29 2022

@author: Landry Fotso (jlandryf@gmail.com)
"""

import cv2,os
import numpy as np
from keras.utils import np_utils

data_path='dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]
label_dict=dict(zip(categories,labels))

data=[]
target=[]

for category in categories:
    folder_path=os.path.join(data_path,category)
    img_names=os.listdir(folder_path)   
    for img_name in img_names:
        img_path=os.path.join(folder_path,img_name)
        img=cv2.imread(img_path)
        try:
            gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
            resized = cv2.resize(gray,(100, 100)) 
            
            data.append(resized)
            target.append(label_dict[category])

        except Exception as e:
            print('Exception:',e)

data=np.array(data)/255.0
data=np.reshape(data,(data.shape[0],100, 100,1))
target=np.array(target)
new_target=np_utils.to_categorical(target)
np.save('data',data)
np.save('target',new_target)








import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten,Dropout
from keras.layers import Conv2D,MaxPooling2D
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

data=np.load('data.npy')
target=np.load('target.npy')

model=Sequential()

model.add(Conv2D(100,(3,3),input_shape=data.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Conv2D(100,(3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(50,activation='relu'))
model.add(Dense(2,activation='softmax'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])

train_data,test_data,train_target,test_target=train_test_split(data,target,test_size=0.1)

checkpoint = ModelCheckpoint(
  'model-{epoch:03d}.model',
  monitor='val_loss',
  verbose=0,
  save_best_only=True,
  mode='auto')

history=model.fit(
  train_data,
  train_target,
  epochs=20,
  callbacks=[checkpoint],
  validation_split=0.2)
print(model.evaluate(test_data,test_target))











from keras.models import load_model
import cv2
import numpy as np
import winsound


model = load_model('model-015.model')
face_clsfr=cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

source=cv2.VideoCapture(0)

labels_dict={1:'with_mask',0:'without_mask'}
color_dict={1:(0,255,0),0:(0,0,255)}

while(True):
    ret,img=source.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=face_clsfr.detectMultiScale(gray,1.3,5)  

    for x,y,w,h in faces:
    
        face_img=gray[y:y+w,x:x+w]
        resized=cv2.resize(face_img,(100,100))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,100,100,1))
        result=model.predict(reshaped)

        label=np.argmax(result,axis=1)[0]
      
        cv2.rectangle(img,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(img,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(
          img, labels_dict[label], 
          (x, y-10),
          cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        """if labels_dict[label] == 'without_mask':
            winsound.Beep(700, 900)"""
        
        
    cv2.imshow('LIVE',img)
    key=cv2.waitKey(1)
    
    if(key==27):
        break
        
cv2.destroyAllWindows()
source.release()









