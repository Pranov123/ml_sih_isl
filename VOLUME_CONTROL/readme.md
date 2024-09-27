# Hand Tracking Volume Control

Welcome to the Hand Tracking Volume Control project! This project uses computer vision to track hand movements and
control the system volume based on the distance between the thumb and index finger. It's a cool and intuitive way to
manage your audio without touching any hardware controls.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Usage](#usage)

## Introduction

This project leverages OpenCV and MediaPipe to detect hand landmarks and control the system's audio volume using the
Pycaw library. The distance between the thumb and index finger is used to set the volume level, making it an innovative
and hands-free approach to volume control.

## Features

- Real-time hand detection and tracking using OpenCV and MediaPipe.
- Volume control based on the distance between thumb and index finger.
- Visual feedback with FPS and volume percentage display.
- Easy to use and extend for other hand gesture-based applications.

## Requirements

- Python 3.x
- OpenCV
- MediaPipe
- Numpy
- Pycaw
- Comtypes

## Usage

1. Run the script

  ```bash
   python name_of_your_file.py
   ```

2. The script will open your webcam and start detecting your hand movements. Adjust the distance between your thumb and
   index finger to control the volume.

3. Press q to exit the application.

## Project Structure

```bash
PROJ_VOLUME_CONTROL/
│
├── HANDTRACKING/
│   └── handtracking_module.py  # Contains the hand tracking module
│
├── main.py  # Main script to run the project
│
├── README.md  # Project documentation  # List of required packages
```
