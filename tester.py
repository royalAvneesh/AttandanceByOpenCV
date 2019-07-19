import cv2
import os
import numpy as np
import faceRecognation as fr
from PIL import  Image

test_img=cv2.imread("C:/Users/Avneesh/PycharmProjects/known_img/na.jpg")
faces_detected,gray_img=fr.faceDetection(test_img)
print("face_detected:",faces_detected)
faces,faceID=fr.labels_for_training_data('C:/Users/Avneesh/PycharmProjects/training')
face_recoginzer=fr.train_classifier(faces,faceID)
face_recoginzer.save('trainingData.yml')
name={0:"deepika",1:"priyanka",2:"salmaan",3:"nipul"}

for face in faces_detected:
    (x,y,w,h)=face
    roi_gray=gray_img[y:y+w,x:x+h]
    label,confidence=face_recoginzer.predict(roi_gray)
    print("confidence:",confidence)
    print("lable:",label)
    fr.draw_rect(test_img,face)
    predicated_name=name[label]
    if (confidence<37):
        continue
    fr.put_text(test_img, predicated_name, x, y)
resized_img=cv2.resize(test_img,(1000,700))
cv2.imshow("face detection tutorial",resized_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

