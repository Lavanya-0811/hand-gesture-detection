import numpy as np

def extract_hand_landmarks(hand_landmarks):
    """
    Extract and normalize hand landmarks for model input.
    
    Args:
        hand_landmarks: MediaPipe hand landmarks object
        
    Returns:
        numpy array of normalized landmarks
    """
    landmarks = []
    
    # Extract x, y, z coordinates for each landmark
    for landmark in hand_landmarks.landmark:
        landmarks.extend([landmark.x, landmark.y, landmark.z])
    
    # Convert to numpy array and normalize
    landmarks = np.array(landmarks)
    landmarks = landmarks.reshape(1, -1)
    
    # Normalize the landmarks
    landmarks = (landmarks - np.mean(landmarks)) / np.std(landmarks)
    
    return landmarks.flatten()

def preprocess_landmarks(landmarks):
    """
    Preprocess landmarks for model input.
    
    Args:
        landmarks: numpy array of landmarks
        
    Returns:
        preprocessed landmarks
    """
    # Ensure landmarks are in the correct shape
    if len(landmarks.shape) == 1:
        landmarks = landmarks.reshape(1, -1)
    
    # Add any additional preprocessing steps here
    return landmarks 