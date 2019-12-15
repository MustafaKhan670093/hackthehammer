import keras
from keras import applications, Model, models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.applications.resnet50 import ResNet50
import cv2 as cv
from sklearn.utils import class_weight
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.callbacks import ReduceLROnPlateau

from keras.preprocessing.image import ImageDataGenerator
import os
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
dataset = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\skin-cancer-mnist-ham10000\HAM10000_metadata.csv")

encoding = {"akiec": 0, "bcc": 1, "bkl": 2, "df": 3, "mel": 4, "nv": 5, "vasc": 6}


labels = dataset.iloc[:, 2].to_numpy()
images = dataset.iloc[:, 1].to_numpy()

labelimg = []



countey = [0 for i in range(7)]
for i in labels:
    temp_label = [0 for i in range(7)]
    wowow = encoding[i]
    temp_label[wowow] = 1
    labelimg.append(temp_label)
images_path = []

base = r"C:\\Users\\adity\\OneDrive\\Desktop\\mnist skin cancer\\skin-cancer-mnist-ham10000\\files\\"

for filename in os.listdir(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\skin-cancer-mnist-ham10000\files"):
    temp_image = cv.imread(base + filename)
    temp_image = cv.resize(temp_image, (75, 100))
    images_path.append(temp_image)
    print(base + filename)

X = np.asarray(images_path)
y = np.asarray(labelimg)

train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2)

print(len(train_y))
print(len(train_X))
print(len(test_y))
print(len(test_X))
train_X,test_X = np.asarray(train_X), np.asarray(test_X)

train_X = train_X/255
train_X = np.asarray(train_X, dtype = "float32")
train_y = np.asarray(train_y, dtype= "float32")

test_X = test_X/255
test_X = np.asarray(test_X, dtype = "float32")
test_y = np.asarray(test_y, dtype= "float32")

print(test_y.shape)
print(test_X.shape)
print(train_X.shape)
print(train_y.shape)


convolution_base = VGG16(weights='imagenet',include_top=False,input_shape= train_X.shape[1:])

res_model= models.Sequential()
res_model.add(convolution_base)
res_model.add(Flatten())
res_model.add(Dropout(0.4))
res_model.add(Dense(500, activation="elu"))
res_model.add(Dropout(0.4))
res_model.add(Dense(250, activation="elu"))
res_model.add(Dropout(0.4))
res_model.add(Dense(100, activation="elu"))
res_model.add(Dropout(0.3))
res_model.add(Dense(7, activation='softmax'))
batch_size=32
print(res_model.summary())


opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)
#learning_rate_reduction = ReduceLROnPlateau(monitor='val_acc',
 #                                           patience=3,
  #                                          verbose=1,
   #                                         factor=0.5,
    #                                        min_lr=0.00001)
res_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])



history = res_model.fit(train_X, train_y,
              batch_size= batch_size,
              epochs= 23,
              validation_data= (test_X, test_y),
              shuffle=True,
              class_weights = class_weights)


print(history.history.keys())
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize init_model for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


res_model.save("wengy.h5")