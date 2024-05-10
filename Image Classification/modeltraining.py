import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
import numpy as np
import random
import os

# Define constants
input_shape = (224, 224, 3)
num_classes = 10
batch_size = 32
num_epochs = 5

# Load MobileNet model without top (classification) layer and pre-trained weights
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=input_shape)

# Add a global average pooling layer and a dense classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the MobileNet base
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Data augmentation for training and validation
train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

# Load and augment the data for training and validation
train_generator = train_datagen.flow_from_directory("C:/Users/majid/trt/HandGesture/cm pro/training",
                                                    target_size=input_shape[:2],
                                                    batch_size=batch_size,
                                                    class_mode='categorical')

test_generator = test_datagen.flow_from_directory("C:/Users/majid/trt/HandGesture/cm pro/testing",
                                                  target_size=input_shape[:2],
                                                  batch_size=batch_size,
                                                  class_mode='categorical')

# Train the model
history = model.fit(train_generator, epochs=num_epochs, validation_data=test_generator)
# Save the trained model
model.save('HandGesture.h5')

# Evaluate the model
evaluation = model.evaluate(test_generator)
print("Test Loss:", evaluation[0])
print("Test Accuracy:", evaluation[1])

# Plot training and validation accuracy over epochs
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
class_indices = train_generator.class_indices

# Print class indices
for class_name, index in class_indices.items():
    print(f"Class '{class_name}' has index {index}")

