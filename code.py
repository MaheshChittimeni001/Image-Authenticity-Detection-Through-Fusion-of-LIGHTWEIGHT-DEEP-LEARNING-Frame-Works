import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Step 1: Set Up Paths for Dataset
data_dir = "dataset"  # Replace with the path to your dataset
train_dir = os.path.join(data_dir, "train")
val_dir = os.path.join(data_dir, "val")
test_dir = os.path.join(data_dir, "test")

# Step 2: Data Preprocessing
def preprocess_data():
    # Data augmentation for training data
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="binary"
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="binary"
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(128, 128),
        batch_size=32,
        class_mode="binary",
        shuffle=False
    )

    return train_generator, val_generator, test_generator

train_generator, val_generator, test_generator = preprocess_data()

# Step 3: Define the Lightweight CNN Model
def build_lightweight_cnn(input_shape=(128, 128, 3)):
    model = Sequential([
        # Convolutional Layer 1
        Conv2D(16, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer 2
        Conv2D(32, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Convolutional Layer 3
        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        # Flatten and Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary Classification
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_lightweight_cnn()
model.summary()

# Step 4: Train the Model
history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=20,
    batch_size=32
)

# Step 5: Evaluate the Model
def evaluate_model():
    test_loss, test_acc = model.evaluate(test_generator)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")

    # Confusion Matrix and Classification Report
    y_true = test_generator.classes
    y_pred = (model.predict(test_generator) > 0.5).astype("int32").flatten()

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_true, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

evaluate_model()

# Step 6: Plot Training and Validation Metrics
def plot_metrics():
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy')
    plt.plot(epochs, val_acc, label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss')
    plt.plot(epochs, val_loss, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.show()

plot_metrics()

# Step 7: Save the Model
model.save("lightweight_cnn_forgery_detection.h5")

# Step 8: Deployment Script (Optional with Streamlit)
def deploy_example():
    import streamlit as st
    from PIL import Image

    st.title("Image Forgery Detection")

    uploaded_file = st.file_uploader("Choose an image to classify...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)

        # Preprocess image
        image = np.array(image.resize((128, 128))) / 255.0
        image = np.expand_dims(image, axis=0)

        # Make prediction
        prediction = model.predict(image)
        if prediction > 0.5:
            st.write("### The image is FORGED")
        else:
            st.write("### The image is AUTHENTIC")

# Uncomment the following line to run the deployment script
# deploy_example()
