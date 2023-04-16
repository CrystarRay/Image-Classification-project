import numpy as np
from keras.applications import MobileNetV2
from keras.applications.mobilenet_v2 import preprocess_input
from keras.utils import img_to_array, load_img



model = MobileNetV2(weights='imagenet')

def classify_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    return MobileNetV2.decode_predictions(preds, top=1)[0][0]