## Overview
This project (SIH 2024) aims to bridge the communication gap between the deaf community and individuals who do not know sign language by providing a robust system to convert spoken language into Indian Sign Language (ISL) and vice versa. The tool facilitates seamless interaction in various scenarios, such as healthcare and public announcements, empowering the deaf community to access critical information and engage fully in society.

## Key Features
- **Speech to ISL Conversion**:
  - Converts spoken language (English/Multiple Indic languages) into ISL gestures using real-time speech recognition and text-to-sign conversion.
  - Ensures accuracy and supports regional ISL variations.
- **Text to ISL Conversion**:
  - Displays ISL gestures corresponding to given text inputs, satisfying use cases like railway announcements.
- **ISL to Text Conversion**:
  - Recognizes ISL gestures captured through a camera and converts them into text. Currently supports letter-based gestures, with word-level gesture recognition under development.

## Technologies Used
- **MediaPipe**: Used for detecting and rendering hand landmarks for text-to-ISL conversion.
- **Sarvam AI**: Enables speech-to-text functionality for spoken language recognition.
- **Python**: Backend implementation for processing inputs and generating outputs.
- **OpenCV**: Used for real-time video feed capture and gesture recognition.
- **Natural Language Processing (NLP)**: Translates recognized speech into text, forming the basis for ISL generation using GenAI.

## Use Cases
1. **Public Announcements**:
   - Converts audio/text announcements at railway stations or other public venues into ISL gestures, ensuring accessibility for the deaf community.
2. **Mobile App for ISL Interpretation**:
   - Enables normal individuals to use their mobile cameras to interpret ISL gestures by deaf individuals.

## Current Status
- **Speech to ISL**:
  - Fully functional, with accurate conversion of spoken content into ISL gestures.
  - Tested and validated for use cases like public announcements.
- **ISL to Text**:
  - Successfully implemented letter-based recognition.
  - Word-level gesture recognition is in progress, with challenges related to complex training on gesture sequences.


## Challenges and Future Work
- **Word-Level Gesture Recognition**:
  - Developing a robust model for recognizing dynamic gesture sequences.
- **Regional ISL Variations**:
  - Incorporating adaptability for regional differences in ISL.

## Full-Stack Project Repository
The complete source code for the integrated full-stack application — including both frontend and backend components — is available at the following GitHub repository:

🔗 [Lingua - GitHub Repository](https://github.com/Team-GenX-H4CK3RS/Lingua)


## Acknowledgements
- **Sarvam AI** for speech recognition.
