# Automated Video Redaction and PII Protection System

A real-time privacy protection system that automatically detects and blurs sensitive information in videos and live webcam streams using **YOLO**, **OpenCV**, and **Gradio**.

## Overview

Sharing videos online often exposes Personally Identifiable Information (PII) such as faces, license plates, ID cards, mobile phones, and computer screens. Manually hiding these details is time-consuming and error-prone.

This project automates the redaction process by detecting sensitive objects in real time and applying Gaussian blur to protect user privacy.

---

## Features

- Face detection using OpenCV Haar Cascade
- Object detection using YOLOv8
- Automatic blurring of:
  - Faces
  - License Plates
  - Mobile Phones
  - Laptop/TV Screens
  - ID Cards
- Supports uploaded video files
- Live webcam redaction
- Object tracking for smooth blur across frames
- Adjustable blur strength and confidence thresholds
- User-friendly Gradio interface

---

## Tech Stack

- Python
- OpenCV
- YOLOv8 (Ultralytics)
- Gradio
- NumPy
- Hugging Face Models

---

## Project Workflow

1. Upload a video or start the webcam.
2. Detect sensitive objects using Haar Cascade and YOLO models.
3. Track detected objects across frames.
4. Apply Gaussian Blur to detected regions.
5. Display or save the processed video.

---

## Installation

Clone the repository

```bash
git clone https://github.com/yourusername/your-repository.git
cd your-repository
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
python mini_project.py
```

---

## Technologies Used

- Python
- OpenCV
- Ultralytics YOLOv8
- NumPy
- Gradio
- Hugging Face

---

## Applications

- CCTV privacy protection
- Law enforcement video anonymization
- Social media content privacy
- Research datasets
- Video conferencing privacy
- Corporate data protection

---

## Future Improvements

- GPU acceleration
- OCR-based text redaction
- Cloud deployment
- Audio PII redaction
- Multi-person tracking
- Additional object detection models
