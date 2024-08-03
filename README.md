# Online-Crowd-Analyzer

## Python Requirements

Make sure to use python 3.12 which was build with the bz2 headers and lzma enabled.
You can download them on debian using `sudo apt-get install libbz2-dev liblzma-dev -y`.

cv2 requires the following packages

- ffmpeg
- libsm6
- libext6
  Install them on debian with `sudo apt-get install ffmpeg libsm6 libxext6  -y`

# Models

The beste detector and tracker found when writing this readme are

- Detector: yolo
  - model: yolov8s.pt
  - confidence: 0.6
- Tracker: centroid
  - maxDissappeared: 40
  - maxDistance: 50

For Rasberry pie's a NCNN version of yolov8n.pt is recommended.
You can use one by setting the argument `ncnn: true` in the config file.


## Coral

sudo apt-get install libhdf5-serial-dev
Add yourself to plugdev group `sudo usermod -aG plugdev [your username]`
