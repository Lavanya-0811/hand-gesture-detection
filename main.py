import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
from hand_utils import extract_hand_landmarks

class ASLRecognizer:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Load the trained model
        try:
            self.model = tf.keras.models.load_model('asl_model.h5')
        except:
            print("Warning: No trained model found. Please train the model first.")
            self.model = None
            
        # Define gesture classes (ASL alphabet)
        self.gesture_classes = [chr(i) for i in range(65, 91)]  # A-Z
        
        # Load ASL sign images
        self.sign_images = {}
        self.load_sign_images()

    def load_sign_images(self):
        """Load ASL sign images for visualization"""
        try:
            for letter in self.gesture_classes:
                img_path = f'sign_images/{letter}.png'
                try:
                    img = cv2.imread(img_path)
                    if img is not None:
                        # Resize image to a standard size
                        img = cv2.resize(img, (100, 100))
                        self.sign_images[letter] = img
                except:
                    print(f"Warning: Could not load sign image for {letter}")
        except:
            print("Warning: Could not load sign images directory")

    def draw_text_with_background(self, frame, text, position, font_scale, color, thickness, bg_color=(0, 0, 0), bg_opacity=0.5):
        """Draw text with a semi-transparent background"""
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
        
        # Draw background rectangle
        x, y = position
        padding = 10
        cv2.rectangle(frame, 
                     (x - padding, y - text_height - padding),
                     (x + text_width + padding, y + padding),
                     bg_color, -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)

    def process_frame(self, frame):
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame and detect hands
        results = self.hands.process(rgb_frame)
        
        # Create a copy of the frame for drawing
        display_frame = frame.copy()
        
        # Add title with modern styling
        self.draw_text_with_background(
            display_frame,
            "ASL Sign Recognition",
            (20, 40),
            1.2,
            (255, 255, 255),
            2
        )
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    display_frame, 
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                if self.model is not None:
                    # Extract hand landmarks for prediction
                    landmarks = extract_hand_landmarks(hand_landmarks)
                    
                    # Make prediction
                    prediction = self.model.predict(np.array([landmarks]))
                    predicted_class = self.gesture_classes[np.argmax(prediction[0])]
                    confidence = np.max(prediction[0]) * 100
                    
                    # Display prediction with confidence
                    self.draw_text_with_background(
                        display_frame,
                        f"Detected Sign: {predicted_class} ({confidence:.1f}%)",
                        (20, 90),
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    
                    # Display ASL sign image if available
                    if predicted_class in self.sign_images:
                        sign_img = self.sign_images[predicted_class]
                        # Place the sign image in the top-right corner
                        h, w = sign_img.shape[:2]
                        display_frame[10:10+h, -w-10:-10] = sign_img
        else:
            # Display message when no hand is detected
            self.draw_text_with_background(
                display_frame,
                "No hand detected",
                (20, 90),
                1.0,
                (0, 0, 255),
                2
            )
        
        # Add instructions
        self.draw_text_with_background(
            display_frame,
            "Press 'Q' to quit",
            (20, display_frame.shape[0] - 20),
            0.7,
            (255, 255, 255),
            2
        )
        
        # Add copyright notice
        copyright_text = "(c) Lav"
        # Calculate text width to position it properly
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_width, text_height), _ = cv2.getTextSize(copyright_text, font, 0.6, 1)
        # Position text in bottom right with padding
        x_pos = display_frame.shape[1] - text_width - 20
        y_pos = display_frame.shape[0] - 10
        self.draw_text_with_background(
            display_frame,
            copyright_text,
            (x_pos, y_pos),
            0.6,
            (200, 200, 200),
            1
        )
        
        return display_frame

def main():
    # Initialize the recognizer
    recognizer = ASLRecognizer()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set window properties
    cv2.namedWindow('ASL Gesture Recognition', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Gesture Recognition', 1200, 800)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = recognizer.process_frame(frame)
        
        # Display the frame
        cv2.imshow('ASL Gesture Recognition', processed_frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 