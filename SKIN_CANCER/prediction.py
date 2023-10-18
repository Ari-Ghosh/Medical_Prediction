from tensorflow import keras
import cv2
import numpy as np
from keras_preprocessing.image import ImageDataGenerator

model = keras.models.load_model("Model/skinDiseaseDetection.h5")
resize1 = 100
resize2 = 75

def preprocess_image(image):
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (resize1, resize2)) / 255
    image = np.array(image)
    image = np.expand_dims(image, axis=-1)
    image = np.expand_dims(image, axis=0)
    return image

def prediction(image):
    image = preprocess_image(image)
    output = model.predict(image)
    output = np.argmax(output,axis=1)
    if output ==0 : 
        return 'actinic keratosis'
    
    elif output ==1 : 
        return 'basal cell carcinoma'
    
    elif output ==2 : 
        return 'dermatofibroma'
    
    elif output ==3 : 
        return 'melanoma'
    
    elif output ==4 : 
        return 'nevus'
    
    elif output ==5 : 
        return 'pigmented benign keratosis'
    
    elif output ==6 : 
        return 'seborrheic keratosis'
    
    elif output ==7 : 
        return 'squamous cell carcinoma'
    
    else : 
        return 'vascular lesion'
