# Indian Sign Language (ISL) to Text/Speech Translation

This project provides a solution to translate Indian Sign Language (ISL) into real-time text and speech, helping bridge the communication gap between the deaf and hard-of-hearing community and the hearing world. It uses computer vision and machine learning to recognize signs and convert them into textual and speech formats in multiple Indian languages.

## Project Overview

The project focuses on the real-time recognition of 35 classes representing (1-9) numbers and (A-Z) alphabet signs from Indian Sign Language (ISL). The model is trained using machine learning to recognize these signs and convert them into corresponding text or speech output. The project leverages OpenCV for image processing and MediaPipe for hand detection and tracking.

### Key Technologies Used:
- **OpenCV**: For capturing images from a webcam and processing them.
- **MediaPipe**: For detecting and recognizing hand landmarks.
- **Machine Learning**: To classify the ISL signs into corresponding characters.

## Project Features

- **Real-time sign recognition** for numbers and alphabets.
- **Text and Speech output** generation in multiple Indian languages.
- **Image collection pipeline** for collecting sign language images.
- **Custom model training** to recognize ISL signs.

## Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- OpenCV
- MediaPipe
- Numpy
- Scikit-learn
- Matplotlib (for plotting, if needed)

Install the required Python libraries using:

```bash
pip install -r requirements.txt
```

## Usage Instructions

Follow the steps below to train the model and test the ISL recognition system.

### 1. Collecting Images of Sign Language

First, you need to collect a dataset of ISL images to train the model.

Run the `preprocessing/collect_imgs.py` file to capture and store images for each sign:

```bash
python preprocessing/collect_imgs.py
```

This script allows you to capture multiple images of hand gestures/signs for each ISL character. Make sure to capture images for each character (A-Z) and numbers (1-9) to cover all 35 classes.

### 2. Creating the Dataset

After collecting the images, you need to draw bounding boxes around the hands and store the hand coordinates for training.

Run the `preprocessing/create_dataset.py` file:

```bash
python preprocessing/create_dataset.py
```

This script processes the collected images, detects hand landmarks using MediaPipe, and draws bounding boxes around the detected hands. The coordinates of these landmarks are stored in arrays, which will later be used to train the model.

### 3. Training the Classifier

Now that we have the dataset prepared, we can proceed to train the machine learning model.

Run the `train_classifier.py` script:

```bash
python train_classifier.py
```

This script trains a machine learning model on the dataset generated in the previous step. The model will learn to recognize different hand signs corresponding to the 35 ISL classes (1-9, A-Z). Once the training is complete, the model is saved for future use.

### 4. Testing the Model

Finally, after training, you can test the model on real-time hand sign inputs.

Run the `inference_classifier.py` file:

```bash
python inference_classifier.py
```

This script captures the webcam feed, detects hand landmarks using MediaPipe, and then uses the trained model to classify the detected signs. The recognized sign is displayed as text and can also be converted to speech using text-to-speech tools.

