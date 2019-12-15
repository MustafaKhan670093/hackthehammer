import keras
from keras import applications, Model, models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.applications.resnet50 import ResNet50
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import pandas as pd
import os
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

df = pd.read_csv(r"C:\Users\adity\OneDrive\Desktop\Programming\GTSRB\gtsrb-german-traffic-sign\Test.csv")

train_X = []
train_y = []

for i in range(0,43):
    n = str(i)
    train_Path = "gtsrb-german-traffic-sign/Train/" + n
    label = [0 for i in range(0, 43)]
    label[i] = 1
    for filename in os.listdir(train_Path):
        img = cv.imread(train_Path + "/" + filename)
        img = cv.resize(img, (32,32))
        print(filename)
        train_X.append(img)
        train_y.append(label)

train_X = np.asarray(train_X)
train_X = train_X/255
train_X = np.asarray(train_X, dtype = "float32")
train_y = np.asarray(train_y, dtype= "float32")


counter = 0
test_X = []
test_y = []
test_Path = "gtsrb-german-traffic-sign/Test"
for filename in os.listdir(test_Path):
        img = cv.imread(test_Path + "/" + filename)
        img = cv.resize(img, (32,32))
        label = [0 for i in range(0, 43)]
        label[df.loc[counter][6]] = 1
        print(filename)
        test_X.append(img)
        test_y.append(label)
        counter += 1

test_X = np.asarray(test_X)
test_X = test_X/255
test_X = np.asarray(test_X, dtype = "float32")
test_y = np.asarray(test_y, dtype= "float32")


print(test_y.shape)
print(train_X.shape())
convolution_base = ResNet50(weights= 'imagenet', include_top=False, input_shape=train_X.shape[1:])
res_model= models.Sequential()
res_model.add(convolution_base)
res_model.add(Flatten())
res_model.add(Dropout(0.2))
res_model.add(Dense(43, activation='softmax'))
batch_size=32

opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

res_model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

history = res_model.fit(train_X, train_y,
              batch_size= batch_size,
              epochs= 2,
              validation_data= (test_X, test_y),
              shuffle=True)


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

# Save model and weights
if not os.path.isdir(r"C:\Users\adity\OneDrive\Desktop\GTSRB"):
    os.makedirs(r"C:\Users\adity\OneDrive\Desktop\GTSRB")
model_path = os.path.join(r"C:\Users\adity\OneDrive\Desktop\GTSRB", "GTSRB_MODEL")
res_model.save(model_path)
print(model_path)



# Score trained model.
scores = res_model.evaluate(test_X, test_y, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])
