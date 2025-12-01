import cv2 as cv
import sys
import os

face_cascade_path = os.path.join(cv.data.haarcascades, "haarcascade_frontalface_default.xml")
face_cascade = cv.CascadeClassifier()

if not face_cascade.load(face_cascade_path):
    print("Error: cannot load Haar cascade:", face_cascade_path)
    sys.exit(1)

# --- Initialize camera ---
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    sys.exit(1)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: cannot read frame")
        break

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv.CASCADE_SCALE_IMAGE
    )

    # Draw rectangles
    for (x, y, w, h) in faces:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show result
    cv.imshow("Face Detection", frame)

    # Press ESC to quit
    if cv.waitKey(1) & 0xFF == 27:
        break

# Cleanup
cap.release()
cv.destroyAllWindows()
