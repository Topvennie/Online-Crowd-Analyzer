import logging

import cv2
from imutils.video import FPS

from config import Config

logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] :: %(message)s")

# Load the configuration file
config = Config.load_from_yaml("config.yaml")

# Input
video = cv2.VideoCapture(config.video)
assert video.isOpened(), "Error: Cannot open video file."

width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps: float = video.get(cv2.CAP_PROP_FPS)
frame_wait = int(1000 / fps) if config.output.show and config.output.realtime else 1

# Output
output = None
if config.output.file is not None:
    fourcc = cv2.VideoWriter.fourcc(*"mp4v")
    output = cv2.VideoWriter(config.output.file, fourcc, fps, (width, height), True)


# Initialize the FPS counter
fps_counter = FPS().start()

ret, frame = video.read()

while ret:
    # Show frame
    if config.output.show:
        cv2.imshow("Real-Time Monitoring/Analysis Window", frame)

        key = cv2.waitKey(frame_wait) & 0xFF
        if key == ord("q"):
            break
        if key == ord("n"):
            fps_counter._numFrames += int(fps * 5)
            video.set(cv2.CAP_PROP_POS_FRAMES, fps_counter._numFrames)
        if key == ord("b"):
            fps_counter._numFrames -= int(fps * 5)
            video.set(cv2.CAP_PROP_POS_FRAMES, fps_counter._numFrames)
        if key == ord("p"):
            cv2.waitKey(0)

    # Save frame
    if output:
        output.write(frame)

    # Go to the next frame
    ret, frame = video.read()
    fps_counter.update()

# Stop the timer and display FPS information
fps_counter.stop()

print(f"Elapsed time: {fps_counter.elapsed():.2f} seconds")
print(f"Approx. FPS: {fps_counter.fps():.2f}")

# Cleanup
video.release()
if output:
    output.release()
if config.output.show:
    cv2.destroyAllWindows()
