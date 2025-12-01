import cv2 as cv
import sys
import time
import numpy as np

NUM_FRAMES = 5        # how many frames to capture
DELAY = 1.0           # delay between frames (seconds)
AUTO_CROP = True      # whether to automatically crop black/empty borders
MODE = "panorama"     # "panorama" (default) or "scans"
OUTPUT_FILE = "panorama_result.jpg"

def capture_images_from_camera(num_frames=NUM_FRAMES, delay=DELAY):
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("Error: could not open camera")
        sys.exit(1)

    images = []
    print("Capturing frames from camera...")

    for i in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Error reading frame from camera")
            continue

        images.append(frame)
        print(f"Captured frame {i+1}/{num_frames}")
        cv.imshow("Captured Frame", frame)
        cv.waitKey(1)
        time.sleep(delay)

    cap.release()
    cv.destroyWindow("Captured Frame")
    return images

def crop_black_borders(pano):
    """
    Crop the largest possible rectangular area that contains ONLY valid pixels.
    Removes all black corners even if panorama shape is irregular.
    """

    print("Cropping black borders...")

    gray = cv.cvtColor(pano, cv.COLOR_BGR2GRAY)

    # Pixels >10 are considered valid
    _, mask = cv.threshold(gray, 10, 255, cv.THRESH_BINARY)

    h, w = mask.shape

    # For each row, find continuous valid horizontal segments
    valid_rect = None
    max_area = 0

    for y1 in range(h):
        for y2 in range(y1 + 1, h):
            # Combine rows y1..y2 (AND)
            region = mask[y1:y2, :]

            # Horizontal projection: a column is valid if all rows inside are valid
            col_valid = np.all(region == 255, axis=0)

            # Now find longest continuous range of True in col_valid
            start = None
            for x in range(w):
                if col_valid[x]:
                    if start is None:
                        start = x
                else:
                    if start is not None:
                        area = (y2 - y1) * (x - start)
                        if area > max_area:
                            max_area = area
                            valid_rect = (start, y1, x - start, y2 - y1)
                        start = None

            # End of row edge
            if start is not None:
                x = w
                area = (y2 - y1) * (x - start)
                if area > max_area:
                    max_area = area
                    valid_rect = (start, y1, x - start, y2 - y1)

    if valid_rect is None:
        print("Unable to find interior rectangle, skipping crop.")
        return pano

    x, y, ww, hh = valid_rect
    cropped = pano[y:y+hh, x:x+ww]

    print("Cropping complete.")
    return cropped



def stitch_images(images):
    """
    Use OpenCV Stitcher to stitch images into panorama.
    """
    print("Stitching images using OpenCV Stitcher...")

    mode_flag = cv.Stitcher_PANORAMA if MODE == "panorama" else cv.Stitcher_SCANS
    stitcher = cv.Stitcher.create(mode_flag)

    status, pano = stitcher.stitch(images)
    if status != cv.Stitcher_OK:
        print("Stitching failed, error code =", status)
        return None

    print("Stitching completed successfully.")
    return pano

def main():
    print("Press ENTER to start capturing frames for panorama...")
    input()

    imgs = capture_images_from_camera()

    if len(imgs) < 2:
        print("Not enough frames to stitch.")
        sys.exit(1)

    pano = stitch_images(imgs)
    if pano is None:
        sys.exit(1)

    if AUTO_CROP:
        pano = crop_black_borders(pano)

    cv.imwrite(OUTPUT_FILE, pano)
    print(f"Panorama saved to {OUTPUT_FILE}")

    cv.imshow("Panorama", pano)
    cv.waitKey(0)
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()
