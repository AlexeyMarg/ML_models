import cv2

img1 = cv2.imread('data/Univ3.jpg')
img2 = cv2.imread('data/Univ2.jpg')
images = [img1, img2]

stitcher = cv2.Stitcher.create(cv2.Stitcher_PANORAMA)
print('Start stiching')
status, panorama = stitcher.stitch(images)
print('Finished stiching')

if status == cv2.Stitcher_OK:
    cv2.imshow('Panorama', panorama)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Error during stitching:", status)