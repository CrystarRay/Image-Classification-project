import tensorflow as tf
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image as keras_image
import numpy as np
from keras.utils import load_img, img_to_array


def classify_image(image_path):
    # Load the pre-trained MobileNetV2 model
    model = MobileNetV2(weights='imagenet')

    # Load and preprocess the image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Perform image classification
    predictions = model.predict(img_array)

    # Decode the predictions into human-readable class names
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Return the top 3 predictions as a list of tuples (class_name, probability)
    return [(pred[1], float(pred[2])) for pred in decoded_predictions]