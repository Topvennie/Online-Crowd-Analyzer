import logging

import cv2
import numpy as np
from cv2.typing import MatLike

from config import Config
from counter.counter import Counter
from number_counter import NumberCounter

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s | [%(levelname)s] :: %(message)s")

# Load the configuration file
config = Config.load_from_yaml("config.yaml")

# Start counter
number_counter = NumberCounter()

counters: list[Counter] = []
counters_show: list[tuple[bool, Counter]] = []

for counter in config.add:
    count = Counter(counter)
    count.run(number_counter.increment)

    if counter.output.show:
        counters_show.append((True, count))

    counters.append(count)

for counter in config.remove:
    count = Counter(counter)
    count.run(number_counter.decrement)

    if counter.output.show:
        counters_show.append((False, count))

    counters.append(count)

# Show the output of the counters that need to show the output
if counters_show:
    cv2.namedWindow("Preview")

    while True:
        frames_add: list[MatLike] = []
        frames_remove: list[MatLike] = []

        for counter in counters_show:
            (show, count) = counter

            (ret, frame) = count.get_frame()

            if not ret:
                counters_show.remove(counter)
                continue

            if show:
                frames_add.append(frame)
            else:
                frames_remove.append(frame)

        if not counters_show:
            break

        # Add empty frames to make the number of frames equal
        deficit = [np.zeros_like(frames_add[0] if len(frames_add) > 0 else frames_remove[0])] * \
            abs(len(frames_add) - len(frames_remove))
        frames_add += deficit if len(frames_remove) > len(frames_add) else []
        frames_remove += deficit if len(frames_add) > len(frames_remove) else []

        # Show all frames
        frame_add = np.hstack(frames_add)
        frame_remove = np.hstack(frames_remove)
        frame = np.vstack([frame_add, frame_remove])

        cv2.imshow("Preview", frame)

        # Parse any potential input
        key = cv2.waitKey(config.show.wait) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("p"):
            cv2.waitKey(0)

        if cv2.getWindowProperty("Preview", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyWindow("Preview")

for counter in counters:
    counter.wait_until_finished()


# TODO: Add buffer
# TODO: Try tensor flow lite for pi
