from tensorflow import keras
import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

model = keras.models.load_model("Model/brain_model.h5")
resize = 200

def preprocess_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (resize, resize)) / 255
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def prediction(image):
    image = preprocess_image(image)
    output = model.predict(image)
    output = np.argmax(output,axis=1)
    if output ==0 : 
        return 'Glioma tumor'
    
    elif output ==1 : 
        return 'Meningioma tumor'
    
    elif output ==2 : 
        return 'No tumor'
    
    else : 
        return 'Pituitary tumor'
