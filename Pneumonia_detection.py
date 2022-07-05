# %%
import os
import pandas as pd
import numpy as np
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import cv2
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import initializers
from tensorflow.keras.callbacks import ModelCheckpoint as MC
from tensorflow.keras.models import load_model

print('Modules importés')



# %%
root = 'data/rsna-pneumonia-detection-challenge/'
train_labels = pd.read_csv(root + 'stage_2_train_labels.csv')
train_labels.sample(5)



# %%    
train_labels['x'] = train_labels['x'].fillna(0)
train_labels['y'] = train_labels['y'].fillna(0)
train_labels['width'] = train_labels['width'].fillna(0)
train_labels['height'] = train_labels['height'].fillna(0)



# %%
class_info = pd.read_csv(root +  'stage_2_detailed_class_info.csv')
class_info.sample(5)


# %%
df = pd.concat([train_labels, class_info['class']], axis=1)
df.sample(5)
sample_trainingdata = df.groupby('class', group_keys=False).apply(lambda x: x.sample(800))
sample_trainingdata["class"].value_counts()

# %%
def readAndReshapeImage(image):
    img = np.array(image).astype(np.uint8)
    ## Resize the image
    res = cv2.resize(img,(128,128), interpolation = cv2.INTER_LINEAR)
    return res


# %%
def générateur_input(Data):
    imageList = []
    classLabels = []

    for index, row in Data.iterrows():
        patientId = row.patientId
        classlabel = row["class"]
        dcm_file = 'data/rsna-pneumonia-detection-challenge/stage_2_train_images/'+'{}.dcm'.format(patientId)
        dcm_data = pydicom.read_file(dcm_file)
        img = dcm_data.pixel_array
        ## Converting the image to 3 channels as the dicom image pixel does not have colour classes wiht it
        if len(img.shape) != 3 or img.shape[2] != 3:
            img = np.stack((img,) * 3, -1)
        imageList.append(readAndReshapeImage(img))
#         originalImage.append(img)
        classLabels.append(classlabel)
    input_Images = np.array(imageList)
    input_Labels = np.array(classLabels)
    enc = LabelBinarizer()
    input_Labels = enc.fit_transform(input_Labels)
#     originalImages = np.array(originalImage)
    return input_Images,input_Labels


# %%
images,labels = générateur_input(sample_trainingdata)



# %%
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=50)

# %%
print('images shape : ' + str(images.shape))
print('labels shape : ' + str(labels.shape))
print('X_train shape : ' + str(X_train.shape))
print('y_train shape : ' + str(y_train.shape))


# %%
