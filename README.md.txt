# AI-Powered Real-Time Object Motion Detection and Tracking for Intelligent Surveillance and Monitoring on Pedestrians

## Overview

This project presents an AI-powered pedestrian detection and tracking system designed for intelligent surveillance and monitoring applications. The system uses the YOLOv8 deep learning model to detect pedestrians in images, videos, and live webcam streams with high accuracy. It helps automate surveillance by reducing manual monitoring and enabling real-time object detection.

---

## Features

- Real-time pedestrian detection
- Pedestrian tracking in videos
- Supports image, video, and webcam input
- High-speed object detection using YOLOv8
- Bounding box visualization
- Confidence score prediction
- Easy-to-use Jupyter Notebook implementation

---

## Technologies Used

- Python
- Jupyter Notebook
- YOLOv8 (Ultralytics)
- OpenCV
- NumPy
- Matplotlib
- PyTorch

---

## Dataset

This project is trained using publicly available pedestrian detection datasets such as:

- Penn-Fudan Pedestrian Dataset
-MOT17 pedestrian Dataset
-Wideperson Dataset
-IITDELHI pedestrian Dataset

> Note: The datasets are not included in this repository because of their large size.

---

## Project Structure

```
AI-Real-Time-Object-Motion-Detection/
│
├── Pedestrian_Detection.ipynb
├── dataset.yaml
├── requirements.txt
├── README.md
├── models/
├── outputs/
├── images/
└── .gitignore
```

---

## Installation

1. Clone the repository

```bash
git clone https://github.com/your-username/your-repository-name.git
```

2. Move to the project directory

```bash
cd your-repository-name
```

3. Install the required libraries

```bash
pip install -r requirements.txt
```

---

## How to Run

1. Open `Pedestrian_Detection.ipynb` in Jupyter Notebook.
2. Install the required packages.
3. Load the YOLOv8 model.
4. Train the model (optional).
5. Run prediction on an image, video, or webcam.
6. View the detected pedestrians in the output.

---

## Results

The model successfully detects pedestrians in:

- Images
- Recorded videos
- Live webcam streams

The output includes:

- Bounding boxes
- Confidence scores
- Real-time detection visualization

---

## Applications

- Smart City Surveillance
- Traffic Monitoring
- Public Safety
- Crowd Monitoring
- Security Systems
- Shopping Malls
- Railway Stations
- Airports
- Parking Areas

---

## Future Enhancements

- Multi-object tracking using DeepSORT or ByteTrack
- Crowd density estimation
- Human activity recognition
- Face anonymization for privacy
- Edge device deployment
- Mobile application integration

---

## Author

**Chandrakala Baswa**

B.Tech Computer Science and Information Technology

Artificial Intelligence and Machine Learning Enthusiast

---

## License

This project is intended for educational and research purposes.