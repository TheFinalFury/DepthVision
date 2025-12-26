import freenect
import cv2
import numpy as np

cv2.namedWindow('Depth')

def pretty_depth(depth):
    np.clip(depth, 0, 2**10 - 1, depth)
    depth >>= 2
    depth = depth.astype(np.uint8)
    return depth

def video_cv(video):
    return video[:, :, ::-1]

def get_depth():
    return pretty_depth(freenect.sync_get_depth()[0])

while 1:
    cv2.imshow('Depth', get_depth())
    if cv2.waitKey(10) == 27:
        break