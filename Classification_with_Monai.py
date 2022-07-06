# %%
import os
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import numpy as np
import cv2
from monai.metrics import ROCAUCMetric
from monai.networks.nets import DenseNet121
from monai.transforms import (
    Activations,
    AddChannel,
    AsDiscrete,
    Compose,
    LoadImage,
    RandFlip,
    RandRotate,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.data import Dataset, DataLoader
from monai.utils import set_determinism

import pandas as pd
import pydicom

from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split


print('Modules importés')



# %%
root = 'data/rsna-pneumonia-detection-challenge/'
train_labels = pd.read_csv(root + 'stage_2_train_labels.csv')
train_labels['x'] = train_labels['x'].fillna(0)
train_labels['y'] = train_labels['y'].fillna(0)
train_labels['width'] = train_labels['width'].fillna(0)
train_labels['height'] = train_labels['height'].fillna(0)
class_info = pd.read_csv(root +  'stage_2_detailed_class_info.csv')
df = pd.concat([train_labels, class_info['class']], axis=1)
sample_trainingdata = df.groupby('class', group_keys=False).apply(lambda x: x.sample(800))
print('Tableau de données initialisé')




# %%
def readAndReshapeImage(image):
    img = np.array(image).astype(np.uint8)
    ## Resize the image
    res = cv2.resize(img,(128,128), interpolation = cv2.INTER_LINEAR)
    return res

def get_data(Data):
    imageList = []
    classLabels = []

    for index, row in Data.iterrows():
        patientId = row.patientId
        classlabel = row["class"]
        dcm_file = 'data/rsna-pneumonia-detection-challenge/stage_2_train_images/'+'{}.dcm'.format(patientId)
        dcm_data = pydicom.read_file(dcm_file)
        img = dcm_data.pixel_array
        imageList.append(readAndReshapeImage(img))
#         originalImage.append(img)
        classLabels.append(classlabel)
    input_Images = np.array(imageList)
    input_Labels = np.array(classLabels)
    enc = LabelBinarizer()
    input_Labels = enc.fit_transform(input_Labels)
#     originalImages = np.array(originalImage)
    return input_Images,input_Labels.astype(float)

print("Défintion des fonctions : \n\n-readAndReshapeImage pour l'obtention et le dimensionnement des données\n\n-get_data pour le chargement d'un jeu de données destinées à l'entraînement du réseau de neurones")



# %%
images,labels = get_data(sample_trainingdata)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=50)
print("Chargement de la donnée terminé")



# %%
train_transforms = Compose([
    AddChannel(),
    ScaleIntensity(),
    RandZoom(min_zoom=0.9, max_zoom=1.1, prob=0.5, keep_size=True),
    ToTensor()
])

val_transforms = Compose([
    AddChannel(),
    ScaleIntensity(),
    ToTensor()
])

print("Définition des transformations sur la donnée en vue de générer les inputs")


# %%
class Sample_Data(Dataset) :

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]

print("Définition de la classe des échantillons d'inputs")



# %%
train_input = Sample_Data(X_train, y_train,train_transforms)
val_input = Sample_Data(X_val,y_val,val_transforms)

train_batch_size = 256
val_batch_size = 64
train_dataloader = DataLoader(train_input, batch_size=train_batch_size)
val_dataloader = DataLoader(val_input,batch_size=val_batch_size)
print("Initialisation de :\n\n-train_dataloader, le dataloader destiné à l'entraînement du réseau\n\nval_dataloader, le dataloader destiné aux calculs de performances de ce réseau")



# %%
model = DenseNet121(
    pretrained = True,
    spatial_dims=2,
    in_channels=1,
    out_channels=3
)
for param in model.parameters():
    param.requires_grad = False

model.class_layers = nn.Sequential(
          nn.ReLU(inplace=True),
          nn.AdaptiveAvgPool2d(output_size=1),
          nn.Flatten(start_dim=1, end_dim=-1),
          nn.Linear(in_features=1024, out_features=3, bias=True)
        )

model = model.to('cpu')
print('Définition du modèle utilisé')



# %%
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    nb_batches = size//train_batch_size
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to('cpu'), y.to('cpu')

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        loss = loss.item()
        print(f"loss : {loss:>7f}    batch : [{batch+1}/{nb_batches+1}]")
        #print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('cpu'), y.to('cpu')
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, correct

print("Initialisation des fonctions d'entraînement et de test")

# %%
epochs = 10
history = []
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.class_layers.parameters(), 0.01)

print("Initalisation des paramètres d'entraînement du modèle")



# %%
print('Entraînement')
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n-------------------------------")
    train(train_dataloader, model, loss_function, optimizer)
    history.append(test(val_dataloader, model, loss_function))
print("Entraînement terminé !")

# %%
torch.save(model, 'Modèles/Monai_pneumonia_detection.pt')

# %%
def plot_loss(history) :
    x = np.linspace(1,epochs,epochs)
    y = []
    for i in range(epochs) :
        y.append(history[i][0])
    plt.plot(x,y)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')

# %%
def plot_accuracy(history) :
    x = np.linspace(1,epochs,epochs)
    y = []
    for i in range(epochs) :
        y.append(history[i][1])
    plt.plot(x,y)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')


# %%
plot_loss(history)

# %%
plot_accuracy(history)


