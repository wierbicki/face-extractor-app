import cv2
import numpy as np
import os

class FaceDetector:
    """
    Face detection class using OpenCV's pre-trained Haar cascades
    """
    
    def __init__(self):
        """Initialize the face detector with pre-trained models"""
        self.face_cascade = None
        self.load_models()
    
    def load_models(self):
        """Load the face detection models"""
        try:
            # Try to load the frontal face cascade
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # Verify the cascade loaded successfully
            if self.face_cascade.empty():
                raise Exception("Failed to load face cascade classifier")
                
        except Exception as e:
            print(f"Error loading face detection models: {e}")
            raise
    
    def detect_faces(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces in an image
        
        Args:
            image: OpenCV image (BGR format)
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size, smaller objects are ignored
            
        Returns:
            List of tuples (x, y, w, h) representing face bounding boxes
        """
        if self.face_cascade is None:
            raise Exception("Face detector not properly initialized")
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale_factor,
            minNeighbors=min_neighbors,
            minSize=min_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        return faces
    
    def detect_faces_with_confidence(self, image, scale_factor=1.1, min_neighbors=5, min_size=(30, 30)):
        """
        Detect faces with additional processing for better accuracy
        
        Args:
            image: OpenCV image (BGR format)
            scale_factor: How much the image size is reduced at each scale
            min_neighbors: How many neighbors each candidate rectangle should have to retain it
            min_size: Minimum possible object size, smaller objects are ignored
            
        Returns:
            List of tuples (x, y, w, h) representing face bounding boxes
        """
        # First attempt with standard parameters
        faces = self.detect_faces(image, scale_factor, min_neighbors, min_size)
        
        # If no faces found, try with more relaxed parameters
        if len(faces) == 0:
            faces = self.detect_faces(image, scale_factor=1.05, min_neighbors=3, min_size=(20, 20))
        
        # Remove overlapping detections
        faces = self._remove_overlapping_faces(faces)
        
        return faces
    
    def _remove_overlapping_faces(self, faces, overlap_threshold=0.3):
        """
        Remove overlapping face detections
        
        Args:
            faces: List of face bounding boxes (x, y, w, h)
            overlap_threshold: Minimum overlap ratio to consider faces as overlapping
            
        Returns:
            Filtered list of face bounding boxes
        """
        if len(faces) <= 1:
            return faces
        
        # Convert to format suitable for NMS (x1, y1, x2, y2)
        boxes = []
        for (x, y, w, h) in faces:
            boxes.append([x, y, x + w, y + h])
        
        boxes = np.array(boxes, dtype=np.float32)
        
        # Calculate areas
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        # Sort by bottom-right y-coordinate
        indices = np.argsort(boxes[:, 3])
        
        keep = []
        while len(indices) > 0:
            # Pick the last index
            last = len(indices) - 1
            i = indices[last]
            keep.append(i)
            
            # Find the largest coordinates for the other bounding boxes
            xx1 = np.maximum(boxes[i, 0], boxes[indices[:last], 0])
            yy1 = np.maximum(boxes[i, 1], boxes[indices[:last], 1])
            xx2 = np.minimum(boxes[i, 2], boxes[indices[:last], 2])
            yy2 = np.minimum(boxes[i, 3], boxes[indices[:last], 3])
            
            # Compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            
            # Compute the intersection over union
            intersection = w * h
            union = areas[i] + areas[indices[:last]] - intersection
            overlap = intersection / union
            
            # Delete all indices from the index list that have IoU greater than threshold
            indices = np.delete(indices, np.concatenate(([last], np.where(overlap > overlap_threshold)[0])))
        
        # Return only the faces that were kept
        filtered_faces = []
        for i in keep:
            x1, y1, x2, y2 = boxes[i]
            filtered_faces.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))
        
        return filtered_faces
    
    def visualize_detections(self, image, faces):
        """
        Draw bounding boxes around detected faces
        
        Args:
            image: OpenCV image (BGR format)
            faces: List of face bounding boxes (x, y, w, h)
            
        Returns:
            Image with bounding boxes drawn
        """
        result_image = image.copy()
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add face number
            cv2.putText(result_image, f'Face', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        return result_image
