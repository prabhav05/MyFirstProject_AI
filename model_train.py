import os
import cv2
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.utils import np_utils 
from keras.layers import  Convolution2D, MaxPooling2D, Activation, concatenate, Dropout,Flatten,Dense
from keras.layers import GlobalAveragePooling2D, GlobalMaxPooling2D
from keras.optimizers import Adam
from keras.models import Sequential

classes={"rock":0,"paper":1,"scissor":2,"none":3}
l=len(classes)

def mapp(value):
    return classes[value]

model=Sequential([
                    Convolution2D(64, 7, activation="relu", padding="same",
                                input_shape=(227,227,3),data_format="channels_last"),
                    MaxPooling2D(2),
                    Convolution2D(128, 3, activation="relu", padding="same"),
                    Convolution2D(128, 3, activation="relu", padding="same"),
                    MaxPooling2D(2),
                    Convolution2D(256, 3, activation="relu", padding="same"),
                    Convolution2D(256, 3, activation="relu", padding="same"),
                    MaxPooling2D(2),
                    Flatten(),
                    Dense(128, activation="relu"),
                    Dense(96,activation="relu"),
                    Dense(64, activation="relu"),
                    Dropout(0.3),
                    Dense(l,activation="softmax")
])
IMG_SAVE_PATH="image_data"
dataset=[]
for directory in os.listdir(IMG_SAVE_PATH):
    path=os.path.join(IMG_SAVE_PATH,directory)
    for image in os.listdir(path):
            img=cv2.imread(os.path.join(path,image))
            img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            img=cv2.resize(img,(227,227))
            dataset.append([img,directory])

data,labels=zip(*dataset)
labels=list(map(mapp,labels))
labels=np_utils.to_categorical(labels)
#print(labels)
#print(np.array(data).shape,np.array(labels).shape)
model.compile(loss="categorical_crossentropy",optimizer=Adam(lr=0.0001),metrics=["accuracy"])
model.fit(np.array(data),np.array(labels),epochs=12)
model.save("rock_paper_scissor_model.h5")

