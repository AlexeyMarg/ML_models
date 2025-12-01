import cv2
import numpy as np

marker = 34

dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)
# Generate marker
markerImage = np.zeros((200, 200), dtype=np.uint8)
markerImage = cv2.aruco.drawMarker(dictionary, marker, 200, markerImage, 1)
cv2.imwrite('marker' + str(marker) + '.png', markerImage)