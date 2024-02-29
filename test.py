import tensorflow as tf
from tensorflow import keras
import numpy as np
import cv2
from keras.models import load_model

facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Use a URL for video capture (replace with your mobile device's IP address and port)
address = "http://192.168.0.102:8080/video"
cap = cv2.VideoCapture(address)

# Set the desired frame width and height
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model('keras_model.h5')

def get_class_name(class_no):
    if class_no == 0:
        return "Sandesh"
    elif class_no == 1:
        return "CaptainA"

while True:
    success, img_original = cap.read()
    faces = facedetect.detectMultiScale(img_original, 1.3, 5)
    for x, y, w, h in faces:
        crop_img = img_original[y:y+h, x:x+h]
        img = cv2.resize(crop_img, (224, 224))
        img = img.reshape(1, 224, 224, 3)
        predictions = model.predict(img)
        class_index = np.argmax(predictions)
        probability_value = np.amax(predictions)
        
        cv2.rectangle(img_original, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(img_original, (x, y-40), (x+w, y), (0, 255, 0), -2)
        cv2.putText(img_original, str(get_class_name(class_index)), (x, y-10), font, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
        
        cv2.putText(img_original, str(round(probability_value*100, 2))+"%", (180, 75), font, 0.75, (255, 0, 0), 2, cv2.LINE_AA)
    
    cv2.imshow("Result", img_original)
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
