# %%
# 1. Import useful libraries

import os
import gc
from IPython.display import clear_output
import matplotlib.pyplot as plt
import torch
import time
import math
import numpy as np
import cv2
from monai.networks.nets import DenseNet201
from monai.transforms import (
    AddChannel,
    Compose,
    RandZoom,
    ScaleIntensity,
    ToTensor,
)
from monai.data import Dataset, DataLoader
import pandas as pd
import pydicom
from sklearn.model_selection import train_test_split


print('Modules imported successfully')










# %%
# 2. Load Data

root = 'data/'


def readAndReshapeImage(image):
    """Resize the image to the right format

    Args:
        image (2D array): a 2D-array of pixels to resize

    Returns:
        res (2D numpy array): the 128x128 resized image with uint8 type
    """    

    img = np.array(image).astype(np.uint8)
    ## Resize the image
    res = cv2.resize(img,(128,128), interpolation = cv2.INTER_LINEAR)
    return res

def fit_transform(Labels) :
    """Mpdify labels list turning arrays into vectors of two numbers [0,1] or [1,0], format required by the loss function

    Args:
        Labels (numpy array): a list of 0 or 1 labels from the input

    Returns:
        new_Labels (numpy array): modified list of [0,1] and [1,0]
    """    

    new_Labels = []
    for i in range (len(Labels)) :
        if Labels[i] == 1 :
            new_Labels.append(np.array([0.,1.]))
        else :
            new_Labels.append(np.array([1.,0.]))
    return new_Labels
        
def get_data(Data):
    """Load data from the dataframe and returns two arrays of formatted inputs

    Args:
        Data (Pandas Dataframe): Dataframe containing the data

    Returns:
       input_Images (numpy array): Array of all images
       input_Labels (numpy array): Array of all labels transformed into vectors
    """  

    imageList = []
    classLabels = []

    for index, row in Data.iterrows():
        patientId = row.patientId
        classlabel = row["Target"]
        dcm_file = root + 'stage_2_train_images/' + '{}.dcm'.format(patientId)
        dcm_data = pydicom.read_file(dcm_file)
        img = dcm_data.pixel_array
        imageList.append(readAndReshapeImage(img))
#         originalImage.append(img)
        classLabels.append(classlabel)
    input_Images = np.array(imageList)
    input_Labels = np.array(classLabels)
    input_Labels = fit_transform(input_Labels)
#     originalImages = np.array(originalImage)
    return input_Images,input_Labels


root = 'data/'

print('Loading Data ...')
train_labels = pd.read_csv(root + 'stage_2_train_labels.csv')
sample_trainingdata = train_labels.groupby('Target', group_keys=False).apply(lambda x: x.sample(4000)) #modify this last number to change the sample size
images,labels = get_data(sample_trainingdata)
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=50)

clear_output()
print("Data loaded successfully")
print(f"Training sample size : {len(X_train)}\nValidation sample size : {len(X_val)}")










# %%
# 3. Define input data and model classes

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


class Sample_Data(Dataset) :

    def __init__(self, image_files, labels, transforms):
        self.image_files = image_files
        self.labels = labels
        self.transforms = transforms

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        return self.transforms(self.image_files[index]), self.labels[index]


train_input = Sample_Data(X_train, y_train,train_transforms)
val_input = Sample_Data(X_val,y_val,val_transforms)

#batch size can be modified if needed
train_batch_size = 512
val_batch_size = 64
train_dataloader = DataLoader(train_input, batch_size=train_batch_size)
val_dataloader = DataLoader(val_input,batch_size=val_batch_size)


model = DenseNet201(
    pretrained = True,
    spatial_dims=2,
    in_channels=1,
    out_channels=2 #for binary classification
)
save_model = DenseNet201(
    pretrained = True,
    spatial_dims=2,
    in_channels=1,
    out_channels=2 #for binary classification
)

#We only train the last dense block and the classifier
for param in model.parameters():
    param.requires_grad = False
model.features.denseblock4 = save_model.features.denseblock4
model.class_layers = save_model.class_layers
del save_model
gc.collect()

model = model.to('cpu')

print('Model and DataLoader classes defined successfully')











# %%
# 4. Train the model

class Timer(object):
    #used to measure process time
    def start(self):
        if hasattr(self, 'interval'):
            del self.interval
        self.start_time = time.time()

    def stop(self):
        if hasattr(self, 'start_time'):
            self.interval = time.time() - self.start_time
            del self.start_time 

timer = Timer()  

def train(dataloader, model, loss_fn, optimizer, scheduler):
    """Run the training of the model

    Args:
        dataloader (DataLoader Class): a DataLoader for the input used by the model
        model (Model Class): the model to train
        loss_fn (two arguments function): Loss function
        optimizer (Optimize class): optimizer used for the gradient descent
        scheduler (Scheduler class): learning rate scheduler
    """
    
    size = len(dataloader.dataset)
    model.train()
    nb_batches = math.ceil(size/train_batch_size)
    for batch, (X, y) in enumerate(dataloader):
        timer.start()
        X, y = X.to('cpu'), y.to('cpu')

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        timer.stop()


        loss = loss.item()
        print(f"loss : {loss:>7f}       batch : [{batch+1}/{nb_batches}]        process time : {int(timer.interval*100)/100}")
    scheduler.step()

def test(dataloader, model, loss_fn):
    """Run a validation test of the model
    Args:
        dataloader (DataLoader Class): a DataLoader for the validation inputs
        model (Model Class): the model to train
        loss_fn (two arguments function): Loss function

    Returns:
       test_loss, accuracy, sensitivity, specificity (numpy arrays): metrics history
    """    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, accuracy, TP, FP, TN, FN = 0, 0, 0, 0, 0, 0 
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to('cpu'), y.to('cpu')
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            accuracy += (pred.argmax(1) == y.argmax(1)).type(torch.float).sum().item()
            TP += ((pred.argmax(1)==1)*(y.argmax(1)==1)).type(torch.float).sum().item() #true positive
            FP += ((pred.argmax(1)==1)*(y.argmax(1)==0)).type(torch.float).sum().item() #false positive
            TN += ((pred.argmax(1)==0)*(y.argmax(1)==0)).type(torch.float).sum().item() #true negative
            FN += ((pred.argmax(1)==0)*(y.argmax(1)==1)).type(torch.float).sum().item() #false negative
    test_loss /= num_batches
    accuracy /= size
    sensitivity = TP/(TP + FN)
    specificity = TN/(TN + FP)
    print(f"Test Error: \n Accuracy: {(100*accuracy):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return test_loss, accuracy, sensitivity, specificity



#Parameters of the training, modify is necessary
epochs = 10
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad == True), 0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

history = []
print('Training :')
for t in range(epochs):
    print(f"Epoch {t+1}/{epochs}\n--------------------------------------------------------------")
    train(train_dataloader, model, loss_function, optimizer,scheduler)
    history.append(test(val_dataloader, model, loss_function))
print("Training finished.")

torch.save(model, 'Pneumonia_classification.pt')











# %%
# 5. Visualize performances

def plot_loss(history) :
    x = np.linspace(2,epochs,epochs-1)
    y = []
    for i in range(1,epochs) :
        y.append(history[i][0])
    plt.plot(x,y)
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    plt.title('Validation Loss')

def plot_accuracy(history) :
    x = np.linspace(1,epochs,epochs)
    y = []
    for i in range(epochs) :
        y.append(history[i][1])
    plt.plot(x,y)
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.title('Validation Accuracy')

def plot_sensitivity(history) :
    x = np.linspace(1,epochs,epochs)
    y = []
    for i in range(epochs) :
        y.append(history[i][2])
    plt.plot(x,y)
    plt.xlabel('epochs')
    plt.ylabel('Sensitivity')
    plt.title('Validation Sensitivity')

def plot_specificity(history) :
    x = np.linspace(1,epochs,epochs)
    y = []
    for i in range(epochs) :
        y.append(history[i][3])
    plt.plot(x,y)
    plt.xlabel('epochs')
    plt.ylabel('Specificity')
    plt.title('Validation Specificity')


#plot_loss(history)
plot_accuracy(history)
#plot_sensitivity(history)
#plot_specificity(history)
