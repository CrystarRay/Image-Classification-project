import os
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model

def classify_image(image_path):
    # Get the absolute path to the H5 file
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "my_custom_model.h5")

    # Load the custom model
    model = load_model(model_path)

    # Load and preprocess the image
    img = load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = img_array.reshape((-1, 784))  # Flatten the image

    # Make the prediction
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return predicted_class