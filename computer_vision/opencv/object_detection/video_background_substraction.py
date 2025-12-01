import cv2 as cv
import numpy as np

# ==========================
# SETTINGS
# ==========================
ALGO = "MOG2"          # "MOG2" or "KNN"
ROI = (100, 100, 400, 300)   # (x, y, width, height)
MOTION_THRESHOLD = 40        # higher = less sensitive to noise
MIN_CONTOUR_AREA = 500       # minimum area of motion to count as object
# ==========================


# Select background subtraction algorithm
if ALGO == "MOG2":
    backSub = cv.createBackgroundSubtractorMOG2(detectShadows=True)
else:
    backSub = cv.createBackgroundSubtractorKNN(detectShadows=True)


# Open webcam
capture = cv.VideoCapture(0)
if not capture.isOpened():
    print("Unable to open camera")
    exit(0)

frame_idx = 0

while True:
    ret, frame = capture.read()
    if not ret:
        print("Unable to read frame")
        break

    frame_idx += 1

    # Draw ROI rectangle
    x, y, w, h = ROI
    roi_frame = frame[y:y+h, x:x+w]

    fgMask = backSub.apply(roi_frame)

    # Threshold to reduce shadows and noise
    _, motionMask = cv.threshold(fgMask, MOTION_THRESHOLD, 255, cv.THRESH_BINARY)

    # Find contours of moving objects
    contours, _ = cv.findContours(motionMask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv.contourArea(cnt)
        if area < MIN_CONTOUR_AREA:
            continue  # skip small objects/noise

        # Bounding box
        cx, cy, cw, ch = cv.boundingRect(cnt)

        # Draw contour (inside ROI)
        cv.drawContours(roi_frame, [cnt], -1, (0, 255, 255), 2)

        # Draw bounding box (inside ROI)
        cv.rectangle(roi_frame, (cx, cy), (cx+cw, cy+ch), (0, 255, 0), 2)

        # Draw label
        cv.putText(roi_frame, "motion", (cx, cy - 5),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Draw ROI boundary on main frame
    cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    # Draw frame counter
    cv.rectangle(frame, (10, 2), (160, 25), (255, 255, 255), -1)
    cv.putText(frame, f"Frame: {frame_idx}", (15, 20),
               cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    # Show results
    cv.imshow("Frame", frame)
    cv.imshow("Motion Mask", motionMask)
    cv.imshow("ROI Motion Detection", roi_frame)

    key = cv.waitKey(1)
    if key == ord('q') or key == 27:
        break

capture.release()
cv.destroyAllWindows()
