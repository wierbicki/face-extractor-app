# Face Extractor

## Repository Information
- **Repository Name**: face-extractor-app
- **Repository Description**: A Streamlit web application for uploading images and automatically extracting detected faces as 200x200 pixel image files

## Overview

Face Extractor is a Streamlit-based web application that automatically detects and extracts faces from uploaded images. The application uses OpenCV's Haar cascade classifiers to identify faces in images and extracts them as standardized 200x200 pixel images with expanded context (including hair and surrounding areas). Users can upload images in common formats (JPG, JPEG, PNG) and download the extracted faces individually or as a batch.

## User Preferences

Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend Architecture
- **Streamlit Framework**: Web interface built with Streamlit for rapid prototyping and deployment
- **Column Layout**: Two-column layout design separating original image display from processing results
- **File Upload Component**: Built-in Streamlit file uploader with format validation
- **Interactive Downloads**: Dynamic download links generated for extracted face images

### Backend Architecture
- **Modular Design**: Separated into three main modules:
  - `app.py`: Main application controller and UI logic
  - `face_detector.py`: Core face detection functionality using OpenCV
  - `utils.py`: Utility functions for image processing and file operations
- **Object-Oriented Face Detection**: `FaceDetector` class encapsulates OpenCV Haar cascade functionality
- **Caching Strategy**: Streamlit resource caching (`@st.cache_resource`) for face detector initialization to improve performance

### Data Processing Pipeline
- **Image Format Handling**: PIL (Pillow) for image loading and format conversion
- **OpenCV Integration**: Computer vision processing using OpenCV's pre-trained models
- **Face Detection Algorithm**: Haar cascade classifiers for frontal face detection
- **Image Standardization**: Extracted faces resized to uniform 200x200 pixel dimensions

### Error Handling and Validation
- **Model Loading Validation**: Checks for successful cascade classifier initialization
- **File Format Validation**: Input validation for supported image formats
- **Graceful Error Recovery**: Exception handling for model loading and image processing failures

## External Dependencies

### Computer Vision Libraries
- **OpenCV (cv2)**: Primary computer vision library for face detection using Haar cascades
- **PIL/Pillow**: Image processing and format conversion
- **NumPy**: Array operations and image data manipulation

### Web Framework
- **Streamlit**: Complete web application framework for UI and deployment

### Pre-trained Models
- **Haar Cascade Classifiers**: OpenCV's built-in pre-trained models for face detection
  - `haarcascade_frontalface_default.xml`: Primary frontal face detection model

### File Processing
- **Base64 Encoding**: For generating downloadable image links
- **IO Operations**: In-memory image buffer handling for file operations
- **ZIP File Creation**: Batch download functionality (referenced but not fully implemented in visible code)

### Python Standard Library
- **os**: File system operations
- **io**: In-memory file operations
- **base64**: Encoding for download links
- **zipfile**: Archive creation for batch downloads