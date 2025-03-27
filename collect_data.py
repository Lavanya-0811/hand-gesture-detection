import cv2
import mediapipe as mp
import numpy as np
import os
from hand_utils import extract_hand_landmarks

class ASLDataCollector:
    def __init__(self):
        # Define gesture classes (ASL alphabet) first
        self.gesture_classes = [chr(i) for i in range(65, 91)]  # A-Z
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Create directories for storing data
        self.data_dir = 'asl_dataset'
        self.create_directories()
        
        # Counter for samples
        self.sample_count = 0
        self.samples_per_class = 100  # Number of samples to collect per class
        
        # State variables
        self.current_class = 0
        self.collecting = False
        self.paused = False

    def create_directories(self):
        """Create necessary directories for storing data"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        for letter in self.gesture_classes:
            class_dir = os.path.join(self.data_dir, letter)
            if not os.path.exists(class_dir):
                os.makedirs(class_dir)

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
            f"ASL Data Collection - {self.gesture_classes[self.current_class]}",
            (20, 40),
            1.2,
            (255, 255, 255),
            2
        )
        
        # Display current progress with modern styling
        progress_text = f"Progress: {self.sample_count}/{self.samples_per_class} samples"
        self.draw_text_with_background(
            display_frame,
            progress_text,
            (20, 90),
            1.0,
            (0, 255, 0),
            2
        )
        
        # Display instructions with modern styling
        instructions = [
            ("Press SPACE to Start/Stop Collection", (20, display_frame.shape[0] - 80)),
            ("Press N for Next Letter", (20, display_frame.shape[0] - 40)),
            ("Press Q to Quit", (20, display_frame.shape[0] - 10))
        ]
        
        for text, pos in instructions:
            self.draw_text_with_background(
                display_frame,
                text,
                pos,
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
        
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks
                self.mp_draw.draw_landmarks(
                    display_frame, 
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                if self.collecting and not self.paused:
                    # Extract hand landmarks
                    landmarks = extract_hand_landmarks(hand_landmarks)
                    
                    # Save the landmarks
                    class_dir = os.path.join(self.data_dir, self.gesture_classes[self.current_class])
                    file_path = os.path.join(class_dir, f'sample_{self.sample_count}.npy')
                    np.save(file_path, landmarks)
                    
                    self.sample_count += 1
                    
                    if self.sample_count >= self.samples_per_class:
                        self.collecting = False
                        self.sample_count = 0
                        print(f"Completed collecting samples for {self.gesture_classes[self.current_class]}")
        
        return display_frame

def main():
    # Initialize the collector
    collector = ASLDataCollector()
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Set window properties
    cv2.namedWindow('ASL Data Collection', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('ASL Data Collection', 1200, 800)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process the frame
        processed_frame = collector.process_frame(frame)
        
        # Display the frame
        cv2.imshow('ASL Data Collection', processed_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord(' '):
            collector.collecting = not collector.collecting
            collector.paused = False
        elif key == ord('n'):
            if collector.current_class < len(collector.gesture_classes) - 1:
                collector.current_class += 1
                collector.sample_count = 0
                collector.collecting = False
                print(f"Moving to next class: {collector.gesture_classes[collector.current_class]}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 