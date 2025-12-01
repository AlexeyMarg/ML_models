import cv2
import numpy as np
import socket

def get_angle(qrcode):
    """Calculate the rotation angle of a QR code based on its polygon coordinates"""
    poly = qrcode.polygon
    #print(poly)
    if len(poly) == 4:
        # Calculate angle using the vector between first two points
        angle = np.arctan2(poly[1].y - poly[0].y, poly[1].x - poly[0].x)
        return angle - np.pi/2  # Adjust by 90 degrees
    else:
        return 0

# Initialize ArUco marker detection
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_50)  # Use 4x4 ArUco dictionary with 50 markers
parameters = cv2.aruco.DetectorParameters_create()

# Initialize video capture from IP camera stream
cap = cv2.VideoCapture('http://192.168.1.92/video.mjpg')

# Set up TCP server for data transmission
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('127.0.0.1', 12345))  # Bind to localhost on port 12345
server.settimeout(5)  # Set timeout for connection attempts

server.listen()  # Start listening for incoming connections

# Wait for client connection
flag = False
while flag is not True:
    try:
        user, address = server.accept()  # Accept incoming connection
        flag = True
    except:
        print('Connection failed')
        #exit()

print('connect')  # Connection established

def draw_frame(img, markerCorners, markerIds):
    """Draw bounding boxes and IDs around detected markers on the image"""
    for i in range(len(markerIds)):
        corners, marker = markerCorners[i], markerIds[i]
        corners = corners.astype('uint32')
        
        # Extract corner coordinates
        point0 = (corners[0][0][0], corners[0][0][1])
        point1 = (corners[0][1][0], corners[0][1][1])
        point2 = (corners[0][2][0], corners[0][2][1])
        point3 = (corners[0][3][0], corners[0][3][1])
        color = (255, 0, 0)  # Blue color for bounding boxes
        
        # Draw bounding box around marker
        cv2.line(img, point0, point1, color, 2)
        cv2.line(img, point1, point2, color, 2)
        cv2.line(img, point2, point3, color, 2)
        cv2.line(img, point3, point0, color, 2)
        
        # Display marker ID near the marker
        cv2.putText(img, str(marker), point0, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(255, 0, 0))

# Main processing loop
while True:
    succsess, img = cap.read()  # Read frame from video stream

    if not succsess:
        print('No data')  # No frame received
        break
    else:
        # Detect ArUco markers in the current frame
        markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(img, dictionary, parameters=parameters)
        
        if markerIds is not None:
            #print(len(markerCorners), len(markerIds))
            draw_frame(img, markerCorners, markerIds)  # Visualize detected markers
        
            # Process each detected marker
            for i in range(len(markerIds)):
                text = str(markerIds[i]) + ', '  # Start with marker ID

                # Calculate center coordinates of the marker
                x_centerPixel = int((markerCorners[i][0][0][0] + markerCorners[i][0][1][0] + 
                                   markerCorners[i][0][2][0] + markerCorners[i][0][3][0]) / 4)
                y_centerPixel = int((markerCorners[i][0][0][1] + markerCorners[i][0][1][1] + 
                                   markerCorners[i][0][2][1] + markerCorners[i][0][3][1]) / 4)
                text += str(x_centerPixel) + ', ' + str(y_centerPixel) + ', '

                # Calculate rotation angle of the marker
                x_delta = markerCorners[i][0][1][0] - markerCorners[i][0][0][0]
                y_delta = markerCorners[i][0][1][1] - markerCorners[i][0][0][1]

                angle = np.arctan2(y_delta, x_delta)  # Calculate angle in radians
                text += str(angle)  # Add angle to data string

                # Send marker data to connected client
                user.send(text.encode('utf-8'))

        # Display the processed frame with marker annotations
        cv2.imshow('image', img)
            
        # Check for exit key (press 'q' to quit)
        k = cv2.waitKey(1)
        if k == ord('q'):
            break

# Clean up resources
cap.release()  # Release video capture
cv2.destroyAllWindows()  # Close all OpenCV windows