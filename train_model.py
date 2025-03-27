import numpy as np
import os
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def load_dataset(data_dir):
    """
    Load the ASL dataset from the specified directory.
    
    Args:
        data_dir: Directory containing the ASL dataset
        
    Returns:
        X: Features (hand landmarks)
        y: Labels (ASL letters)
    """
    X = []
    y = []
    
    # Get all class directories
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    for class_dir in class_dirs:
        class_path = os.path.join(data_dir, class_dir)
        
        # Get all sample files for this class
        sample_files = [f for f in os.listdir(class_path) if f.endswith('.npy')]
        
        for sample_file in sample_files:
            # Load the sample
            sample_path = os.path.join(class_path, sample_file)
            landmarks = np.load(sample_path)
            
            X.append(landmarks)
            y.append(ord(class_dir) - ord('A'))  # Convert letter to index (0-25)
    
    return np.array(X), np.array(y)

def create_model(input_shape, num_classes):
    """
    Create a neural network model for ASL recognition.
    
    Args:
        input_shape: Shape of input data
        num_classes: Number of gesture classes
        
    Returns:
        Compiled model
    """
    model = models.Sequential([
        layers.Dense(256, activation='relu', input_shape=input_shape),
        layers.Dropout(0.3),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(64, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history):
    """
    Plot the training history.
    
    Args:
        history: Training history object
    """
    plt.figure(figsize=(12, 4))
    
    # Plot accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

def main():
    # Load the dataset
    print("Loading dataset...")
    X, y = load_dataset('asl_dataset')
    
    if len(X) == 0:
        print("Error: No data found in the dataset directory.")
        print("Please run collect_data.py first to collect training data.")
        return
    
    print(f"Loaded {len(X)} samples")
    
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Create and compile the model
    print("Creating model...")
    model = create_model(input_shape=(X.shape[1],), num_classes=26)
    
    # Train the model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        epochs=50,
        batch_size=32,
        validation_data=(X_val, y_val)
    )
    
    # Plot training history
    print("Plotting training history...")
    plot_training_history(history)
    
    # Save the model
    print("Saving model...")
    model.save('asl_model.h5')
    
    # Evaluate the model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(X_val, y_val)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    print("\nTraining completed!")
    print("Model saved as 'asl_model.h5'")
    print("Training history plot saved as 'training_history.png'")

if __name__ == "__main__":
    main() 