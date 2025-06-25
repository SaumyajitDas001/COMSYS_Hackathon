
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image as PILImage
import os
import sys

label_map = {0: "Male", 1: "Female"}

def predict_gender(img_path, model_path="gender_classification_model.h5"):
    model = tf.keras.models.load_model(model_path)
    img = image.load_img(img_path, target_size=(128, 128))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Display image
    PILImage.open(img_path).show()

    prediction = model.predict(img_array)
    pred = int(prediction[0][0] > 0.5)
    print(f"Predicted Gender: {label_map[pred]}")

    os.remove(img_path)
    print(f"Deleted image: {img_path}")

# Example usage:
# predict_gender("example/sample.jpg")
