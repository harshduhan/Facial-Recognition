# Importing Libraries
import cv2
import numpy as np
import face_recognition
from PIL import Image, ImageOps
from PIL import Image as im
import keras
 
# System Requirements
import sys
import os
from datetime import datetime
from time import sleep
 
 
# Bulk Loading the Images
path = "C:\School XII\py files\images"
images = []
personName = []
myList = os.listdir(path)
 
# Classifying Image Names
for cu_img in myList:
    current_Img = cv2.imread(f'{path}/{cu_img}')
    images.append(current_Img)
    personName.append(os.path.splitext(cu_img)[0])
print("Database: ", personName)
 
 
# Training Faces
def faceEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList
 
 
 
# Finished encoding known faces
encodeListKnown = faceEncodings(images)
print("\nAll Encodings Complete!")
 
print("\nInitializing Webcam...");
sleep(0.25)
 
 
# Defining the Attendance attribute
def attendance(name):
    with open(r"C:\Users\Dell\Downloads\Attendance.csv", 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
 
        # Adding Names to the (.csv) file
        if name not in nameList:
            time_now = datetime.now()
            tStr = time_now.strftime('%H:%M:%S')
            dStr = time_now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{tStr},{dStr}')
 
 
# Initializing the Webcam
cap = cv2.VideoCapture(0)
 
vid = cv2.VideoCapture(0)
while (True):
 
    ret, frame = vid.read()
    cv2.imshow('frame', frame)
 
    if cv2.waitKey(1) & 0xFF == ord('m'):
        cv2.imwrite(r"C:\Users\Dell\Pictures\Camera Roll\1.jpg",frame)
        break
 
vid.release()
 
def teachable_machine_classification(img):
    model = keras.models.load_model(r"C:\Users\Dell\Downloads\keras_model.h5")
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)
    image = img
    size = (224, 224)
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1
    data[0] = normalized_image_array
    prediction = model.predict(data)
    label = np.argmax(prediction)
    if np.argmax(prediction) == 1:
        print('Mask')
    else:
        print('No Mask')
    return label
 
 
new = im.open(r"C:\Users\Dell\Pictures\Camera Roll\1.jpg")
size = (224, 224)
new.resize(size)
x = teachable_machine_classification(new)
print(x)
 
 
# Making the Rectangle Element
while True:
    ret, frame = cap.read()
    faces = cv2.resize(frame, (0, 0), None, 0.25, 0.25)
    faces = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
 
    facesCurrentFrame = face_recognition.face_locations(faces)
    encodesCurrentFrame = face_recognition.face_encodings(faces, facesCurrentFrame)
 
    # Detecting Current Face
    for encodeFace, faceLoc in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
        faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
 
        matchIndex = np.argmin(faceDis)
 
        # Outputting Names
        if matches[matchIndex]:
            name = personName[matchIndex].upper()
            y1, x2, y2, x1 = faceLoc
            y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(frame, name, (x1 + 6, y2 - 6), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)
            attendance(name)
 
    # Quitting the Program
    cv2.imshow("Camera", frame)
    if cv2.waitKey(33) == ord('x'):
        break
 
cap.release()
cv2.destroyAllWindows()
