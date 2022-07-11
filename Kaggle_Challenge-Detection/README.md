# General Description  

This file contains a classic commented pipeline of medical imaging detection. This algorithm was designed as part of the Kaggle RSNA pneumonia detection challenge, the goal is to diagnose and locate pneumonia using a lung scan. The ouput of the neural network is a box, which locates the disease on the lung if a pnuemonia is detected, and surrounds the whole image if not.  
This pipeline does not use a specialized library python for medical deeplearning, since detection is rarely used in medicine. Refer to Classification or Segmentation examples to use these libraries.
  
![alt text](Images\Scan_Example.png "Scan Example")
  
On this example, we can see a pneumonia on the right lung of the patient, recognizable by a characteristic opacity on the scan  

---

# 1. Libraries  

We use several libraries and packages that need to be installed upstream :
- Torchvision package
- Pycocotools
- PyTorch
- Pydicom
- Pandas
- Matplotlib
- Numpy
- Sklearn

## > Torchvision package

torchvision is a package that can be downloaded from github on the link below, using pycocotools. It contains a framework for training a neural network of detection.  
Link : https://github.com/pytorch/vision

## > Pydicom

Pydicom is the library used to open files containing medical data, and to extract and process this data.  
Link : https://pypi.org/project/pydicom/

---

# 2. Required trainig data  

The data we use for training is provided by Kaggle, it comes from the RSNA pneumonia detection challenge. It needs to be downloaded and stored in a folder named *data* to be read automatically by the python script.

---

# 3. Python file description  

The Classification file is divided into 5 segments which can be compiled separately for a better understanding, and simplify script modifications.

1. Import useful libraries
2. Load Data
3. Define input data and model classes
4. Visualize an example of object detection
5. Train the model



## > Import useful libraries  
First segment is simply the import of the libraries presented above.

## > Load Data
This segment defines different functions to simplify the reading of the data. 
After this, we load the csv file using pandas and generate sample of training and evaluation. It is possible to change the size of the desired sample by modifying this line :
```python
sample_data = labels_csv.groupby('Target', group_keys=False).apply(lambda x: x.sample(500))
```
The *500* term designate the number of images per class we want to select for the training, knowing that 20% of them will be used for validation.

## > Define input data and model classes
Using torchvision framework, we define the transformations to apply to the inputs. This allows us to define classes for our datasets and dataloaders. This is where it is possible to modifiy the train and validation batch size that will be used by the model :
```python
batch_size = 8
train_dl = DataLoader(train_ds, batch_size, shuffle=True, collate_fn=utils.collate_fn)
val_dl = DataLoader(val_ds, batch_size*2, collate_fn=utils.collate_fn)
```
After this, we define our model. Here, we use a Faster-RCNN with ResNet50 as backbone pre-trained model, and we re-train the last layers of the network : the box predictor. Regarding the images we want to classify, it might be relevant to change the layers that will be trained.

## > Visualize an example of object detection
This segment offers an overview of the data used for training, and the corresponding bounding boxes.  
  
![alt text](Images\Box_examples.png "Box Examples")  
  
## > Train the_model
After the settings of our parameters, it is time to start the training. 
It is possible to change the parameters of the training, like the number of epochs or the learning rate, by modifying these lines :
```python
num_epochs = 5
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)
```
At the end of the training, a model will be saved under the name *Pneumonia_detection.pt* in your repertory.