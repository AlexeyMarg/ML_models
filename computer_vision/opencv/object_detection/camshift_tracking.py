import cv2 as cv
import numpy as np

# Mouse callback — for selecting ROI to track

drawing = False
ix, iy = -1, -1
rx, ry, rw, rh = 0, 0, 0, 0
roi_selected = False

def select_roi(event, x, y, flags, param):
    global ix, iy, rx, ry, rw, rh, drawing, roi_selected

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y

    elif event == cv.EVENT_MOUSEMOVE:
        if drawing:
            rx, ry = min(ix, x), min(iy, y)
            rw, rh = abs(x - ix), abs(y - iy)

    elif event == cv.EVENT_LBUTTONUP:
        drawing = False
        rx, ry = min(ix, x), min(iy, y)
        rw, rh = abs(x - ix), abs(y - iy)
        roi_selected = True


cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: cannot open camera")
    exit()

cv.namedWindow("Frame")
cv.setMouseCallback("Frame", select_roi)

roi_hist = None
track_window = None

term_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)

print("Draw a rectangle on the object to track, then press ENTER...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    display = frame.copy()
    if drawing or roi_selected:
        cv.rectangle(display, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)

    cv.imshow("Frame", display)
    key = cv.waitKey(10)

    if key == 13 and roi_selected:   # ENTER
        break
    if key == 27:   # ESC
        cap.release()
        cv.destroyAllWindows()
        exit()


roi = frame[ry:ry + rh, rx:rx + rw]
hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)

mask = cv.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))

roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

track_window = (rx, ry, rw, rh)

print("Tracking started! Press ESC to exit.")


while True:
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    back_proj = cv.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    ret, track_window = cv.CamShift(back_proj, track_window, term_criteria)

    pts = cv.boxPoints(ret)
    pts = np.int32(pts)

    cv.polylines(frame, [pts], True, (0, 255, 255), 3)

    cv.imshow("Frame", frame)
    cv.imshow("Back Projection", back_proj)

    key = cv.waitKey(1)
    if key == 27:  # ESC
        break

cap.release()
cv.destroyAllWindows()
