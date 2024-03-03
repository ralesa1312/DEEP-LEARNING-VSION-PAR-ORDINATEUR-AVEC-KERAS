import cv2
from keras.models import model_from_json
import numpy as np


# Utilisation de l'API model après entrainement et test 
json_file = open("emotiondetector.json","r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)

#Les interconnexions sont à extraires aussi
model.load_weights('emotiondetector.h5')
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade=cv2.CascadeClassifier(haar_file)

#redimensioner l'image avec chaque pixels
#Tsara ra atao anaty tableau histoire manatsara calcul des matrices
def exctractfeature(image):
	feature = np.array(image)
	feature = feature.reshape(1,48,48,1)
	return feature/255.0

#Initier CV2 et la capture 
webcam = cv2.VideoCapture(0)
labels = {0 : 'Tezitra', 1 : 'Kivy', 2 : 'Matahotra', 3 : 'Faly', 4 : 'Tsy misy fihetsehampo', 5 : 'Malahelo', 6 : 'Gaga'}
while True:
    i,im = webcam.read()
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(im,1.3,5)
    #Bien encadrer l'images 
    #Doc officiel open-cv
    try:
        for (p,q,r,s) in faces:
            image = gray[q:q+s,p:p+r]
            cv2.rectangle(im,(p,q),(p+r,q+s),(255,0,0),2)
            image = cv2.resize(image,(48,48))
            img = exctractfeature(image)
            pred = model.predict(img)
            prediction_label = labels[pred.argmax()]
            cv2.putText(im, '% s' %(prediction_label), (p-10, q-10), cv2.FONT_HERSHEY_COMPLEX_SMALL,2,(0,0,255))
        cv2.imshow('output',im)
        cv2.waitKey(27)
    except cv2.error:
        pass
        
