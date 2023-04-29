import os
import numpy as np
from keras.utils import load_img, img_to_array
from keras.models import load_model

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def classify_image(image_path):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_dir, "my_custom_model.h5")

    model = load_model(model_path)

    img = load_img(image_path, target_size=(28, 28), color_mode="grayscale")
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize the image
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])

    return class_names[predicted_class]