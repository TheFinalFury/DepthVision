import cv2
import numpy as np
import freenect
import threading
import time

# ------------------------
# Parameters
# ------------------------
MIN_DEPTH = 500
MAX_EFFECTIVE_DEPTH = 1500
DEPTH_STEP = 50
MIN_AREA = 10000

# ------------------------
# Shared state
# ------------------------
depth_raw_shared = None
frame_lock = threading.Lock()
running = True

# ------------------------
# Depth helpers
# ------------------------
def get_depth_raw():
    return freenect.sync_get_depth()[0]  # uint16

def pretty_depth(depth_raw):
    depth = depth_raw.copy()
    np.clip(depth, 0, 1023, depth)
    depth >>= 2
    return depth.astype(np.uint8)

# ------------------------
# Thread: Kinect capture ONLY
# ------------------------
def depth_capture_thread():
    global depth_raw_shared, running

    while running:
        depth = get_depth_raw()
        depth[depth > MAX_EFFECTIVE_DEPTH] = 0

        with frame_lock:
            depth_raw_shared = depth

        time.sleep(0.002)  # yield to USB stack

# ------------------------
# Start capture thread
# ------------------------
capture_thread = threading.Thread(
    target=depth_capture_thread,
    daemon=True
)
capture_thread.start()

# ------------------------
# MAIN THREAD: processing + display
# ------------------------
last_frame = None

while True:
    with frame_lock:
        if depth_raw_shared is not None:
            last_frame = depth_raw_shared.copy()

    if last_frame is None:
        time.sleep(0.01)
        continue

    depth_raw = last_frame

    # Visualization
    depth_vis = pretty_depth(depth_raw)
    vis = cv2.applyColorMap(
        cv2.normalize(depth_vis, None, 0, 255, cv2.NORM_MINMAX),
        cv2.COLORMAP_JET
    )

    # Depth quantization
    valid = (depth_raw >= MIN_DEPTH) & (depth_raw <= MAX_EFFECTIVE_DEPTH)
    quantized = np.full(depth_raw.shape, -1, np.int16)
    quantized[valid] = (depth_raw[valid] - MIN_DEPTH) // DEPTH_STEP

    max_label = quantized.max()
    if max_label >= 0:
        all_contours = []

        for label in range(max_label + 1):
            mask = (quantized == label).astype(np.uint8) * 255

            if cv2.countNonZero(mask) < MIN_AREA:
                continue

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for cnt in contours:
                if cv2.contourArea(cnt) < MIN_AREA:
                    continue
                all_contours.append(cnt)

        # Draw blobs
        num_blobs = len(all_contours)
        for i, cnt in enumerate(all_contours):
            h = int(120 * (i / max(1, num_blobs - 1)))
            hsv = np.uint8([[[h, 255, 255]]])
            color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            cv2.drawContours(vis, [cnt], -1, tuple(int(c) for c in color), -1)

    cv2.imshow("Filled blobs", vis)

    if (cv2.waitKey(1) & 0xFF) == 27:
        running = False
        break

cv2.destroyAllWindows()
