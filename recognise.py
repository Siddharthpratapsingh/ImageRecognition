import numpy as np
import cv2
from pymongo import MongoClient
import os

Client = MongoClient()
db=Client["Database"]
collection=db["Master"]
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

name=raw_input("Enter your name");
password=raw_input("Enter your password");
data = db.collection.find({'name':name,'password':password})
na=0
if data:
    os.chdir(str(name))
    cap = cv2.VideoCapture(0)
    rec = cv2.createLBPHFaceRecognizer();
    rec.load("trainingdata.yml")
    image=[]
    i=[]
    image = cv2.imread(str(name)+'.8.jpg')
    sample=0
    font=cv2.cv.InitFont(cv2.cv.CV_FONT_HERSHEY_COMPLEX_SMALL,5,1,0,4)
    while 1:
        ret, img = cap.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
		gray,
		scaleFactor=3,
		minNeighbors=5,
		minSize=(30, 30)
		#flags = cv2.CV_HAAR_SCALE_IMAGE
	)
        
        for (x,y,w,h) in faces:
           sample=sample+1
           cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
           na,conf=rec.predict(gray[y:y+h,x:x+w])
           cv2.imwrite(str(name)+".jpg", gray[y:y+h, x:x+w])
        
        cv2.imshow('img',img)
        cv2.waitKey(1)
        if sample > 4:
            break
        
        i=cv2.imread(str(name)+".jpg")
        if(image.any()==i.all()):
            break
        else:
            print "Wrong person/Show your face"
            continue
            
  
    
        if cv2.waitKey(1) == ord('q'):
            break
    
    cv2.destroyAllWindows()
