from ultralytics import YOLO
import cv2
import os.path

if os.path.exists('yolov8n.pt'):
    model = YOLO('../yolov8n.pt')
else:
    model = YOLO('yolov8n.pt')


'''
#yolov8n - nano
#yolov8m - medium
#yolov8l - large

'''

results = model('images/1.png', show=True)
cv2.waitKey(0)