
from model import create_model
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_dir = "dataset/train"
val_dir = "dataset/val"

# Image Data Generator
train_gen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(train_dir, target_size=(128, 128), batch_size=32, class_mode='binary')
val_data = val_gen.flow_from_directory(val_dir, target_size=(128, 128), batch_size=32, class_mode='binary')

# Load and train model
model = create_model(model_type="both")
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(train_data, epochs=10, validation_data=val_data)

# Save model
model.save("gender_classification_model.h5")
