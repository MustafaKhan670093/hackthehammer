import keras 
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt

#Invasive Ductal Carcinoma (IDC)
# yes for 1 no for 0
#histopathology --> tissue samples taken for breast cancer

directory = r'C:/Users/amrut/Desktop/Mammograms/mammogram-images/'
labels = []
image_data = []
for filename in os.listdir(directory):
   
    main_path = os.path.join(directory,filename + "/")
    
    
    for subfilename in os.listdir(main_path):
        class_path = os.path.join(main_path,subfilename+"/")
       
        for image in os.listdir(class_path):
            image_path = os.path.join(class_path,image)
            print(image_path)
            label = int(image_path[-5])
            extracted_image = cv2.imread(image_path)
            print(extracted_image.shape)
            extracted_image = cv2.resize(extracted_image, (50,50))

            image_data.append(extracted_image)
            labels.append([label])


            

image_as_np = np.array(image_data)
image_as_np  = image_as_np/255
#print(image_as_np)
#image_collection = np.array(image_data)
#image_collection  = np.vstack(image_data)



final_labels = np.asarray(labels, dtype = np.float32)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(image_as_np, final_labels, test_size=0.15)








from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Conv2D, MaxPool2D, MaxPooling2D, Dense, Dropout, Activation, Flatten
from keras.models import Sequential

from keras.applications.vgg16 import VGG16
conv_base = VGG16(weights='imagenet',include_top=False,input_shape= X_train.shape[1:])
model = Sequential()


model.add(conv_base)
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
model.summary()

model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-4),
          loss='binary_crossentropy',
          metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size= 128,epochs= 3,validation_data= (X_test, y_test))



"""print(history.history.keys())
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
plt.show(block=True)
plt.interactive(False)"""

model.save("mammogram.h5")
#model.save_weights("mammogram.h5")
        


#loop for directory file
   


