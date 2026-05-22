from ultralytics import YOLO
import cv2
import cvzone
import os.path
import math

classNames = ['Excavator', 'Gloves', 'Hardhat', 'Ladder', 'Mask', 'NO-Hardhat', 'NO-Mask', 'NO-Safety Vest', 
              'Person', 'SUV', 'Safety Cone', 'Safety Vest', 'bus', 'dump truck', 'fire hydrant', 'machinery', 'mini-van', 
              'sedan', 'semi', 'trailer', 'truck and trailer', 'truck', 'van', 'vehicle', 'wheel loader']

myColor = (0, 0, 255)

cap = cv2.VideoCapture('ppe-1.mp4')

model = YOLO('ppe.pt')

while True:
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            #cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2-x1, y2-y1
            cvzone.cornerRect(img, (x1, y1, w, h))
            
            # confidence
            confidence = math.ceil(box.conf[0]*100)/100

            # classes
            cls = int(box.cls[0])

            currentClass = classNames[cls]

            if currentClass in ['NO-Hardhat', 'NO-Mask', 'NO-Safety Vest']:
                myColor = (0, 0, 255)
            elif currentClass in ['Hardhat', 'Mask', 'Safety Vest']:
                myColor = (0, 255, 0)
            else:
                myColor = (255, 0, 0)

            cvzone.putTextRect(img, f'{classNames[cls]} {confidence}', (max(0,x1), max(0, y1-10)), scale=1, thickness=1,
                            colorB=myColor, colorT=(255, 255, 255), colorR=myColor)
            cv2.rectangle(img, (x1, y1), (x2, y2), color=myColor, thickness=3)

    cv2.imshow('Image', img)
    cv2.waitKey(1)
