# Face Recognition Attendance System

This project captures real-time video from a webcam and automatically
marks attendance using face recognition.\
It identifies registered individuals, logs their name along with the
current timestamp, and stores the attendance in a CSV file named with
the current date.

## Features

-   Real-time face detection using a webcam\
-   Automatic face recognition using pre-encoded images\
-   Attendance marking with timestamps\
-   CSV output stored per day\
-   Simple and easy-to-understand Python code structure

## Technologies Used

-   Python\
-   OpenCV\
-   face_recognition Library\
-   NumPy\
-   CSV for data logging

## Project Link

GitHub Repository:
https://github.com/EmanIqbal18/Face-Recognition-Attendance-system

## How It Works

1.  Load images of known individuals and generate encodings.\
2.  Capture video frames from the webcam.\
3.  Detect faces and compare them with known encodings.\
4.  Log attendance for recognized faces once per session.\
5.  Save output in a CSV file for the current date.

## How to Run

1.  Install dependencies:

        pip install opencv-python face_recognition numpy

2.  Place known faces inside the **Images** folder.\

3.  Run the script:

        python attendance.py

4.  Press **q** to exit the camera window.
