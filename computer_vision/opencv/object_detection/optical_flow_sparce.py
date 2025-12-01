import cv2 as cv
import numpy as np

# Parameters for corner detection (good features to track)
feature_params = dict(
    maxCorners=200,
    qualityLevel=0.3,
    minDistance=7,
    blockSize=7
)

# Parameters for Lucas-Kanade optical flow
lk_params = dict(
    winSize=(15, 15),
    maxLevel=2,
    criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03)
)

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()

ret, old_frame = cap.read()
if not ret:
    exit()

old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)

# Detect initial corners
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Random colors for tracks
color = np.random.randint(0, 255, (200, 3))

# Mask for drawing trajectories
mask = np.zeros_like(old_frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # Calculate optical flow (Lucas–Kanade)
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    if p1 is None:
        # Re-detect features if lost
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue

    # Keep only good points
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    # Draw tracks
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel().astype(int)
        c, d = old.ravel().astype(int)

        mask = cv.line(mask, (a, b), (c, d), color[i].tolist(), 2)
        frame = cv.circle(frame, (a, b), 4, color[i].tolist(), -1)

    img = cv.add(frame, mask)

    cv.imshow("Sparse Optical Flow (Lucas-Kanade)", img)

    key = cv.waitKey(1)
    if key == 27:   # ESC
        break
    if key == ord('r'):
        # Reset tracking
        mask = np.zeros_like(frame)
        p0 = cv.goodFeaturesToTrack(frame_gray, mask=None, **feature_params)

    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)

cap.release()
cv.destroyAllWindows()
