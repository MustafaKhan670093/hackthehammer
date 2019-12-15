import keras
from keras.models import load_model
import cv2 as cv
import numpy as np
import os
from keras.preprocessing import image

model = load_model(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\goose.h5")
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

n = r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\skin-cancer-mnist-ham10000\files"

#for filename in os.listdir(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\skin-cancer-mnist-ham10000\files"):
image = cv.imread(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\skin-cancer-mnist-ham10000\files\ISIC_0024329.jpg")
image = cv.resize(image, (75,100))
images_path = []
images_path.append(image)
images_path = np.asarray(images_path)
images_path = images_path / 255
test_X = np.asarray(images_path, dtype="float32")

    #cv.imshow("wobble", image)
    #cv.waitKey(0)  # waits until a key is pressed
    #cv.destroyAllWindows()  # destroys the window showing image
encoding = {0: "akiec", 1:"bcc", 2:"bkl", 3 : "df", 4: "mel", 5: "nv", 6:"vasc"}
result = model.predict_classes(test_X)
print(encoding[int(result)])
