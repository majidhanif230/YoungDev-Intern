# Project Title: Hand Gesture Recognition with Real-time and Flask-based Web Application

## Overview:
This project utilizes computer vision and deep learning techniques for hand gesture recognition. It includes two main components: a real-time implementation using OpenCV and TensorFlow, and a Flask-based web application for gesture recognition.

## Real-time Implementation:
### Requirements:
- OpenCV
- TensorFlow
- cvzone (HandTrackingModule and ClassificationModule)
- pyttsx3
- numpy

### Usage:
1. Run the real-time implementation script (`texttovoice.py`).
2. The script captures video from the default camera and performs hand gesture recognition using a pre-trained model.
3. The recognized gestures are displayed in real-time, and the corresponding text is spoken using the text-to-speech engine.

## Flask-based Web Application:
### Requirements:
- Flask
- OpenCV
- TensorFlow
- cvzone (ClassificationModule)
- numpy

### Usage:
1. Run the Flask web application script (`app.py`).
2. Navigate to `http://localhost:5000` in a web browser.
3. Upload an image containing a hand gesture to receive the recognition result.
4. The web application uses the pre-trained model to predict the gesture and displays the corresponding text.

## Files:
- `texttovoice.py`: Real-time implementation script.
- `app.py`: Flask-based web application script.
- `HandGesture.h5`: Pre-trained model for gesture classification.
- `hello.txt`: Text file containing class names for gesture recognition.

## Folder Structure:
- `templates`: Contains HTML templates for the Flask application.

## Dependencies:
Ensure that the required dependencies are installed before running the scripts. Use the following command to install dependencies:
```bash
pip install opencv-python tensorflow pyttsx3 Flask numpy cvzone
```

## Troubleshooting:
If you encounter any issues, refer to the error messages or raise an issue on the GitHub repository.

## Acknowledgments:
- The project utilizes the `cvzone` library for hand tracking and classification modules.
- The pre-trained model (`HandGesture.h5`) is used for gesture classification.

## Notes:
- This project is for educational and demonstrative purposes.
- Feel free to customize and enhance the code based on your requirements.

## License:
Free to Use

## Author:
Majid Hanif

## Contributing:
1. Fork the repository.
2. Create a new branch for your changes.
3. Commit your changes and push to your branch.
4. Submit a pull request.


Enjoy gesture recognition with this project!
