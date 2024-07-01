# Vehicle Detection and Speed Estimation

This project implements a vehicle detection and speed estimation system using computer vision techniques with OpenCV in Python. The system processes video footage to detect vehicles, estimate their speed, and capture images of vehicles exceeding a specified speed limit.

## Key Features

1. **Vehicle Detection**: Uses background subtraction and contour detection to identify moving vehicles in the video.

2. **Speed Estimation**: Calculates the speed of each detected vehicle based on its movement between frames.

3. **Speeding Vehicle Capture**: Automatically captures and saves images of vehicles exceeding a predefined speed limit.

4. **Vehicle Counting**: Counts the number of vehicles passing a specific line in the frame.

## How It Works

1. The script reads frames from a video file.
2. It applies background subtraction to isolate moving objects.
3. Contour detection is used to identify potential vehicles.
4. The position of each vehicle is tracked across frames to estimate its speed.
5. If a vehicle's speed exceeds the limit when crossing a predefined line, an image of the vehicle is captured and saved.
6. The total count of vehicles is updated and displayed on the video frame.

## Key Components

- `cv2.bgsegm.createBackgroundSubtractorMOG()`: For background subtraction.
- `cv2.findContours()`: To detect vehicle contours.
- `calculate_speed()`: Function to estimate vehicle speed.
- `capture_vehicle_image()`: Function to capture and save images of speeding vehicles.

## Limitations and Assumptions

- Assumes a stationary camera.
- Speed calculation is based on a fixed distance assumption and may need calibration for accurate results.
- Works best with vehicles moving perpendicular to the camera view.

## Potential Improvements

- Implement a more robust vehicle tracking algorithm.
- Calibrate the speed calculation based on known distances in the video frame.
- Add multi-camera support for more accurate speed and position estimation.


