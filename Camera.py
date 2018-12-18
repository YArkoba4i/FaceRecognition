import numpy as np
import cv2 
import pickle
import ctypes

face_cascade = cv2.CascadeClassifier('e:\Developming\GitHubRepo\Python\Faces\Faces\cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('e:\Developming\GitHubRepo\Python\Faces\Faces\cascades\data\haarcascade_eye.xml')
smile_cascade = cv2.CascadeClassifier('e:\Developming\GitHubRepo\Python\Faces\Faces\cascades\data\haarcascade_smile.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face-trainner.yml")

labels = {"person_name": 1}
with open("face-labels.pickle", 'rb') as f:
	og_labels = pickle.load(f)
	labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)

name=''

while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    print(faces)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w] #(ycord_start, ycord_end)
        roi_color = frame[y:y+h, x:x+w]
        
        id_, conf = recognizer.predict(roi_gray)

        print(conf)
        print(id_)
        if conf>=4 and conf <= 85:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[id_]
            print(name)
            color = (255, 255, 255)
            stroke = 2
            cv2.putText(frame, name, (x,y), font, 1, color, stroke, cv2.LINE_AA)

        else:
            print(id_)
            print(conf)
            cv2.rectangle(frame, (x,y), (end_cord_X, end_cord_Y), color, stroke)
            img_item = "intruder.png"
            cv2.imwrite(img_item, roi_color)
            ctypes.windll.user32.LockWorkStation()

        color = (0, 255, 0)
        stroke = 2
        end_cord_X = x+w
        end_cord_Y = y+h
     

    cv2.imshow('frame', frame)
    if cv2.waitKey(20) & 0xFF == ord("q") or name == 'yarko':
        print(name)
        break

cap.relese()
cv2.destroyAllWindows()
