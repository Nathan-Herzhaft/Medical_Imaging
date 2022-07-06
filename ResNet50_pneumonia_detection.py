# %%
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import math
import sys
import logging

import os
import pydicom
import imageio
from tqdm.auto import tqdm, trange
import cv2

import torch
from PIL import Image
import torchvision.transforms as T
from PIL import ImageDraw
from torch.utils.data import random_split, DataLoader, Dataset
from torch import tensor
from torchvision.utils import make_grid
import torchvision
#from engine import train_one_epoch, evaluate
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from vision.references.detection import utils as utils
from vision.references.detection.engine import train_one_epoch, evaluate

print('Modules importés')




# %%
train_path = 'data/rsna-pneumonia-detection-challenge/stage_2_train_images/'
test_path = 'data/rsna-pneumonia-detection-challenge/stage_2_test_images/'
path = 'data/rsna-pneumonia-detection-challenge'

labels_csv = pd.read_csv(os.path.join(path,'stage_2_train_labels.csv'))

labels_csv.x.fillna(0, inplace=True)
labels_csv.y.fillna(0, inplace=True)
labels_csv.width.fillna(1023, inplace=True)
labels_csv.height.fillna(1023, inplace=True)

labels_csv['x_max'] = labels_csv['x']+labels_csv['width']
labels_csv['y_max'] = labels_csv['y']+labels_csv['height']



print('Import du tableau de données')
labels_csv.head()



# %%
def parse_one_annot(box_coord, filename):
   boxes_array = box_coord[box_coord["patientId"] == filename][["x", "y",        
   "x_max", "y_max"]].values
   
   return boxes_array

def dicom_to_array(image_path):
    dcm_data = pydicom.read_file(image_path)
    im = dcm_data.pixel_array
    return im


class RSNA(Dataset):
    def __init__(self, path, box_coord, transforms=None):
        self.path = path
        self.box_coord = box_coord
        self.transforms = transforms
        self.imgs = sorted(os.listdir(path))

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.path, self.box_coord['patientId'][idx] + '.dcm')
        img = dicom_to_array(img_path)
        #img = Image.open(img_path).convert("RGB")
        #img = img.resize((1024, 1024))
        box_list = parse_one_annot(self.box_coord, 
        self.box_coord['patientId'][idx])
        boxes = torch.as_tensor(box_list, dtype=torch.float32)
        num_objs = len(box_list)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:,
        0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transforms is not None:
                img = self.transforms(img)
        return img, target
    def __len__(self):
          return len(self.box_coord['patientId'])

print('Définition de la classe du Dataset à partir des donénes')

# %%
def train_tfms():
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(T.ToTensor())
   return T.Compose(transforms)

def val_tfms():
   transforms = []
   # converts the image, a PIL image, into a PyTorch Tensor
   transforms.append(T.ToTensor())
   return T.Compose(transforms)

print('Transformations à appliquer aux images')

# %%
np.random.seed(0)
sample_data = labels_csv.groupby('Target', group_keys=False).apply(lambda x: x.sample(1200))
msk = np.random.rand(len(sample_data)) < 0.9

train_df = sample_data[msk].reset_index()
val_df = sample_data[~msk].reset_index()
train_ds = RSNA(train_path, train_df, transforms=train_tfms())
val_ds = RSNA(train_path, val_df, transforms=val_tfms())


print('Initialisation des classes de training et validation')
print(f'Taille échantillon training :{len(train_ds)}, Taille échantillon validation : {len(val_ds)}')

batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=utils.collate_fn)
val_dl = DataLoader(val_ds, batch_size*2, collate_fn=utils.collate_fn)



# %%
def draw_bounding_box(img, label_boxes):
  all_imgs = []
  for i in range(img.shape[0]):        
      image = img[i,:,:,:]
      image = image.squeeze(0)
      im = Image.fromarray(image.mul(255).byte().numpy())
      draw = ImageDraw.Draw(im)
      labels = label_boxes[i]['boxes']
      for elem in range(len(labels)):
        draw.rectangle([(labels[elem][0], labels[elem][1]),
        (labels[elem][2], labels[elem][3])], 
        outline ="red", width = 10)
      all_imgs.append(np.array(im))
  all_imgs = np.array(all_imgs)
  return T.ToTensor()(all_imgs)

def show_batch(dataloader):
    for images, labels in dataloader:
        image = draw_bounding_box(torch.stack(images), labels)
        image = image.permute(1,2,0).mul(255).byte().numpy()
        fig, ax = plt.subplots(figsize=(16, 16), nrows=2, ncols=3)
        gs1 = gridspec.GridSpec(3, 4)
        gs1.update(wspace=0.030, hspace=0.030) # set the spacing between axes. 
        id = 0
        for i in range(2):
            for j in range(3):
                ax[i,j].set_title('Exemple :')
                ax[i,j].imshow(image[id], cmap=plt.cm.bone)
                id = id + 1
        
        plt.show()
        break

print('Exemples de données')
show_batch(train_dl)



# %%
def get_model(num_classes): #à modifier pour utiliser un autre modèle
   # load an object detection model pre-trained on COCO
   model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
   for param in model.parameters():
    param.requires_grad = False
   # get the number of input features for the classifier
   in_features = model.roi_heads.box_predictor.cls_score.in_features
   # replace the pre-trained head with a new on
   model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
   
   return model

num_classes = 2

model = get_model(num_classes)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)


print('Initialisation du modèle ResNet50, avec un classifier à entraîner')




# %%
print('Entraînement du modèle')

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_dl, 'cpu', epoch, print_freq=1)
    lr_scheduler.step()
    evaluate(model, val_dl, device='cpu')

print('Entraînement terminé')
torch.save(model, 'Modèles/ResNet50_pneumonia_detection.pt')

# %%
