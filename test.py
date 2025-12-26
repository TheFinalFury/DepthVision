import freenect
import cv2
import numpy as np

cv2.namedWindow('Depth')

i = 0

def pretty_depth(depth):
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def get_depth():
    return pretty_depth(freenect.sync_get_depth()[0])

while True:
    cv2.imshow('Depth', get_depth())

    key = cv2.waitKey(10) & 0xFF
    if key == 27:  # ESC key
        break

cv2.destroyAllWindows()