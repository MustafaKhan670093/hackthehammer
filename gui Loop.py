import keras
from keras.models import load_model
import os
import cv2 as cv
import numpy as np
import PySimpleGUI as sg
model1 = load_model(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\models\goose.h5")
model1.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

model2 = load_model(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\models\colorectal.h5", compile = False)
model2.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


model3 = load_model(r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer\models\mammogram.h5", compile = False)
model3.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

encoding = {0: "Actinic keratoses", 1:"basal cell carcinoma", 2:"benign keratosis-like lesions", 3 : "dermatofibroma", 4: "melanoma ", 5: "melanocytic nevi ", 6:"vascular lesions"}
encoding1 = {0: "tumor", 1: "stroma", 2: "complex", 3: "lympho", 4: "debris", 5: "mucosa", 6: "adipose", 7: "empty"}
encoding2 = {0: "benign", 1: "malignant"}



sg.change_look_and_feel('DarkAmber')	# Add a touch of color
layout = [[sg.Text('Insert a model in the first text field corresponding to what area of the body you are looking at')],
            [sg.Text('Insert the local filepath of the image that you want classified')],
            [sg.Text('model'), sg.InputText("model", key = 'model')],
            [sg.Text('filepath'), sg.InputText("filepath", key = 'file')],
            [sg.Button('Ok'), sg.Button('Cancel')]]

window = sg.Window('C2').Layout(layout)

# Create the Window
# Event Loop to process "events" and get the "values" of the inputs
while True:
    event, values = window.read()
    print(event)
    if event in (None, 'Cancel'):	# if user closes window or clicks cancel
        break
    if event == "Ok":
        print("hgo")
        base = r"C:\Users\adity\OneDrive\Desktop\mnist skin cancer"
        pick_model = values['model']
        print(pick_model)
        #filepath = input("enter the filepath")
        process = os.path.join(base, values['file'])
        print(process)
        if pick_model == "skin":
            image = cv.imread(process)
            image = cv.resize(image, (75, 100))
            images_path = []
            test = cv.resize(image, (200,200))
            cv.imshow("skin visual", test)
            images_path.append(image)
            images_path = np.asarray(images_path)
            images_path = images_path / 255
            image = np.asarray(images_path, dtype="float32")
            result = model1.predict_classes(image)
            print(encoding[int(result)])
            sg.Popup("this is" + " " + encoding[int(result)])
        elif pick_model == "colorectal":
            image = cv.imread(process)
            image = cv.resize(image, (50, 50))
            images_path = []
            test = cv.resize(image, (200,200))
            cv.imshow("colorectal visual", test)
            images_path.append(image)
            images_path = np.asarray(images_path)
            images_path = images_path / 255
            image = np.asarray(images_path, dtype="float32")
            result = model2.predict_classes(image)
            print(encoding1[int(result)])
            sg.Popup("this is" + " " + encoding1[int(result)])
        elif pick_model == "mammogram":
            image = cv.imread(process)
            image = cv.resize(image, (50, 50))
            images_path = []
            test = cv.resize(image, (200,200))
            cv.imshow("mammogram visual", test)
            images_path.append(image)
            images_path = np.asarray(images_path)
            images_path = images_path / 255
            image = np.asarray(images_path, dtype="float32")
            result = model3.predict_classes(image)
            print(encoding2[int(result)])
            sg.Popup("this is" + " " + encoding2[int(result)])


window.close()
