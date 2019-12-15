import keras
from keras import applications, Model, models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.applications.resnet50 import ResNet50
import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator
import os
from sklearn.model_selection import train_test_split


import tensorflow as tf
sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

directory = r'C:/Users/amrut/Desktop/Colorectal Cancer/Kather_texture_2016_image_tiles_5000/Kather_texture_2016_image_tiles/'
labels = []
image_data = []
for filename in os.listdir(directory):

    main_path = os.path.join(directory, filename + "/")
    #print(main_path)


    path_label = int(filename[1])-1

    for subfilename in os.listdir(main_path):
        class_path = os.path.join(main_path, subfilename)
        #print(class_path)
        temp = cv2.imread(class_path)
        temp = cv2.resize(temp, (50, 50))
        image_data.append(temp/255)
        labels.append(path_label)






image_data= np.asarray(image_data,dtype=np.float32)
labels= np.asarray(labels,dtype = np.float32)
#label_dict = {"":"","":"","":"","":"","":"","":"","":""}

from keras.utils import to_categorical
encoded_labels = to_categorical(labels)



X_train, X_test, y_train, y_test = train_test_split(image_data, encoded_labels, test_size=0.2, shuffle=True)

from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.models import Sequential

model = Sequential()

model.add(Conv2D(filters = 128, kernel_size = (5,5),padding = 'same',activation ='relu', input_shape = (50,50,3)))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'same',activation ='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(256,activation = "relu"))
model.add(Dense(64,activation = "relu"))
model.add(Dense(32,activation = "relu"))

model.add(Dense(8, activation = "softmax"))

model.compile(optimizer = "Adam" , loss = "categorical_crossentropy", metrics=["accuracy"])

datagen = ImageDataGenerator(
        rotation_range=0.5,
        zoom_range = 0.5,
        width_shift_range=0.5,
        height_shift_range=0.5,
        horizontal_flip=True,
        vertical_flip=True)

datagen.fit(X_train)
model.fit_generator(datagen.flow(X_train,y_train, batch_size=200),
                              epochs = 4, validation_data = (X_test,y_test), steps_per_epoch=500)


"""from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.models import Sequential

from keras.applications.vgg16 import VGG16
conv_base = VGG16(weights='imagenet',include_top=False,input_shape= X_train.shape[1:])
model = Sequential()
model.add(conv_base)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(8, activation='sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-5),
          loss='categorical_crossentropy',
          metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size= 128,epochs= 3,validation_data= (X_test, y_test),verbose = 1)
"""
"""
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
"""
model.save("colorectal.h5")
