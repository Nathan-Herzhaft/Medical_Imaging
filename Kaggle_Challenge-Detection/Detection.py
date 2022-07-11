# %%
# 1. Import useful libraries

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import pydicom
from tqdm.auto import tqdm, trange
import torch
from PIL import Image
import torchvision.transforms as T
from PIL import ImageDraw
from torch.utils.data import random_split, DataLoader, Dataset
from torch import tensor
from torchvision.utils import make_grid
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from vision.references.detection import utils as utils
from vision.references.detection.engine import train_one_epoch, evaluate

print('Modules imported successfully')










# %%
# 2. Load Data

root = 'data/'

def read_image(image_path):
    """Read an image for dicom file path

    Args:
        image_path (string) : dicome file path

    Returns:
        im (_type_) : 2D array of the image
    """    
    dcm_data = pydicom.read_file(image_path)
    im = dcm_data.pixel_array
    return im

def get_box_coord(Dataframe, Id):
    """Get the 4 coordinates of the box considered

    Args:
        Dataframe (pandas Dataframe): dataset
        Id (str): patient id indexed in the dataset

    Returns:
        boxes_array (array) : array of the 4 box coordinates
    """    
    boxes_array = Dataframe[Dataframe["patientId"] == Id][["x", "y",        
    "x_max", "y_max"]].values
   
    return boxes_array

labels_csv = pd.read_csv(os.path.join(root,'stage_2_train_labels.csv'))

#By default, set the box for healthy patient to the whole frame
labels_csv.x.fillna(0, inplace=True)
labels_csv.y.fillna(0, inplace=True)
labels_csv.width.fillna(1023, inplace=True)
labels_csv.height.fillna(1023, inplace=True)

labels_csv['x_max'] = labels_csv['x']+labels_csv['width']
labels_csv['y_max'] = labels_csv['y']+labels_csv['height']

np.random.seed(0)
sample_data = labels_csv.groupby('Target', group_keys=False).apply(lambda x: x.sample(500))
mask = np.random.rand(len(sample_data)) < 0.8
train_df = sample_data[mask].reset_index()
val_df = sample_data[~mask].reset_index()

print("Data loaded successfully")
print(f"Training sample size : {len(train_df)}\nValidation sample size : {len(val_df)}")










# %%
# 3. Define input data and model classes

def train_tfms():
    """Define transformations to apply to training dataset

    Returns:
        (Transform Compose) : sequence of transformations
    """    
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

def val_tfms():
    """Define transformations to apply to validation dataset


    Returns:
        (Transform Compose) : sequence of transformations
    """    
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    return T.Compose(transforms)

class RSNA(Dataset):
    def __init__(self, path, Dataframe, transforms=None):
        self.path = path
        self.Dataframe = Dataframe
        self.transforms = transforms
        self.imgs = sorted(os.listdir(path))

    def __getitem__(self, idx):
        # load images and bounding boxes
        img_path = os.path.join(self.path, self.Dataframe['patientId'][idx] + '.dcm')
        img = read_image(img_path)
        #img = Image.open(img_path).convert("RGB")
        #img = img.resize((1024, 1024))
        box_list = get_box_coord(self.Dataframe, 
        self.Dataframe['patientId'][idx])
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
          return len(self.Dataframe['patientId'])


train_ds = RSNA(root + 'stage_2_train_images', train_df, transforms=train_tfms())
val_ds = RSNA(root + 'stage_2_train_images', val_df, transforms=val_tfms())

batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=utils.collate_fn)
val_dl = DataLoader(val_ds, batch_size*2, collate_fn=utils.collate_fn)

def get_model(num_classes):
    """Load the pretrained model

    Args:
        num_classes (int): dimension of the output

    Returns:
        (Neural Network Class) : pretrained loaded model with reinitialized predictor
    """    
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

print('Model and DataLoader classes defined successfully')










# %%
# 4. Visualize an example of object detection

def draw_bounding_box(img, label_boxes):
    """Return an image with apparent bounding box

    Args:
        img (array): stacked images to draw
        label_boxes (labels in DataLaoder): box description

    Returns:
        (Tensor): all images with drawn boxes
    """    
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
    """Sample of images with bounding box to visualize an example

    Args:
        dataloader (Dataloader): dataloader from which we extract the sample
    """    
    for images, labels in dataloader:
        image = draw_bounding_box(torch.stack(images), labels)
        image = image.permute(1,2,0).mul(255).byte().numpy()
        fig, ax = plt.subplots(figsize=(12,9), nrows=2, ncols=3)
        gs1 = gridspec.GridSpec(3, 4)
        gs1.update(wspace=0.030, hspace=0.030) # set the spacing between axes. 
        id = 0
        for i in range(2):
            for j in range(3):
                ax[i,j].imshow(image[id], cmap=plt.cm.bone)
                id = id + 1
        
        plt.show()
        break

print('Data Examples')
show_batch(train_dl)










# %%
# 5. Train the model

#Parameters of the training, modify is necessary
num_epochs = 5
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

print('Training :')
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_dl, 'cpu', epoch, print_freq=1)
    lr_scheduler.step()
    evaluate(model, val_dl, device='cpu')
print("Training finished.")

torch.save(model, 'Pneumonia_detection.pt')