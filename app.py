# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping

# Step 1: Data Preprocessing
# Function to load and preprocess images for CNN
def load_images_from_directory(directory, img_size=(224, 224)):
    images = []
    labels = []
    
    for folder in os.listdir(directory):
        label = folder  # Label is the folder name
        folder_path = os.path.join(directory, folder)
        
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)  # Load image
            img = cv2.resize(img, img_size)  # Resize image to match model input
            images.append(img)
            labels.append(label)
    
    images = np.array(images)
    labels = np.array(labels)
    
    # Normalize images to values between 0 and 1
    images = images.astype('float32') / 255.0
    
    return images, labels

# Step 2: Image Augmentation (for CNN model)
def augment_data(train_images):
    # Using ImageDataGenerator to apply random transformations to the images
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    return datagen.flow(train_images)

# Step 3: Model Architecture for SVM
# Function to extract features from images and train an SVM model
def train_svm_model(features, labels):
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    # Initialize the SVM classifier
    svm_model = SVC(kernel='linear', probability=True)
    
    # Train the SVM model
    svm_model.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = svm_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Accuracy of SVM model: {accuracy*100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    
    return svm_model

# Step 4: Model Architecture for CNN
# Function to define and train a CNN model
def build_and_train_cnn(train_images, train_labels, val_images, val_labels):
    # CNN model architecture
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(2, activation='softmax')  # Output layer (2 classes: Tumor or No Tumor)
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Apply early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    
    # Train the model
    model.fit(train_images, train_labels, epochs=20, batch_size=32, validation_data=(val_images, val_labels), callbacks=[early_stop])
    
    return model

# Step 5: Model Evaluation (CNN)
def evaluate_cnn_model(model, test_images, test_labels):
    # Evaluate the model on the test set
    test_loss, test_acc = model.evaluate(test_images, test_labels)
    print(f"Test accuracy: {test_acc*100:.2f}%")
    
    # Make predictions
    predictions = model.predict(test_images)
    predicted_labels = np.argmax(predictions, axis=1)
    
    print("Classification Report for CNN:")
    print(classification_report(test_labels, predicted_labels))

# Step 6: Main Execution Function
def main():
    # Load the images and labels (ensure proper directory structure: `tumor` and `no_tumor` folders)
    images, labels = load_images_from_directory('data/brain_tumor', img_size=(224, 224))
    
    # Split data into train and test sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    
    # Train the SVM Model (Extract features and train)
    # In practice, we would first extract features using a method like HOG or CNN, but here we use raw pixels
    svm_model = train_svm_model(X_train, y_train)
    
    # Build and train the CNN model
    cnn_model = build_and_train_cnn(X_train, y_train, X_test, y_test)
    
    # Evaluate the CNN model
    evaluate_cnn_model(cnn_model, X_test, y_test)
    
# Run the main function to execute the entire process
if __name__ == "__main__":
    main()
