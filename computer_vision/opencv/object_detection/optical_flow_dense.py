import cv2 as cv
import numpy as np

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

# Read the first frame
ret, frame1 = cap.read()
if not ret:
    exit()

prev_gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)

hsv = np.zeros_like(frame1)
hsv[..., 1] = 255  # Full saturation for colorful flow visualization

while True:
    ret, frame2 = cap.read()
    if not ret:
        break

    gray = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)

    # Calculate dense optical flow
    flow = cv.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5,   # pyr scale
        3,     # levels
        15,    # winsize
        3,     # iterations
        5,     # poly_n
        1.2,   # poly_sigma
        0
    )

    # Flow components
    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])

    # Hue = angle of flow
    hsv[..., 0] = ang * 180 / np.pi / 2

    # Value = magnitude of flow
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

    # Convert HSV → BGR for display
    rgb_flow = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow("Dense Optical Flow (Farneback)", rgb_flow)

    # Keyboard
    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

    prev_gray = gray

cap.release()
cv.destroyAllWindows()
