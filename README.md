# Simple_Lane_Detection

This project demonstrates lane detection in a video using OpenCV and computer vision techniques.

## Installation

1. Clone the repository:`https://github.com/Basfore05/Simple_Lane_Detection.git`
2. Install OpenCV: `pip install opencv-python`

## Usage

1. Place the video file you want to process in the same directory as the script.
2. Update the `test2.mp4` filename in the script to the name of your video file.
3. Run the script: `python lane_detection.py`

## Description

The script performs the following steps:

1. Reads a video frame-by-frame.
2. Applies the Canny edge detection algorithm to detect edges in the frame.
3. Extracts the region of interest (ROI) from the edge-detected frame.
4. Applies the Hough transform to detect lines in the ROI.
5. Averages and extrapolates the detected lines to estimate the lane boundaries.
6. Draws the estimated lane lines on the original frame.
7. Displays the processed frame with the lane lines.

## Dependencies

- Python 3.x
- OpenCV
- NumPy

## License

This project is licensed under the [MIT License](LICENSE).
