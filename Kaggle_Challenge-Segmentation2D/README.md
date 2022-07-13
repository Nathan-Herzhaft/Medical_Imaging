# General Description  

This file contains a classic commented pipeline of medical imaging segmentation. This algorithm was designed as part of the Kaggle Ultrasound Nerve Segmentation challenge, the goal is to identify and segment nerve collections using ultrasound images.  

On the example below, we can see the ultrasound image on the left and the segmented ouput on the right. The image comes from an opensource notebook on the challenge page, this pipeline does not provide data visualization.
  
![alt text](Images\Image_Example.png "Image Example")  
  
  

---

# 1. Libraries  

We use several libraries and packages that need to be installed upstream :
- Monai
- PyTorch

## > Monai

Monai proposes a framework to support the creation of a computervision pipeline in the medical field. It is an extension of PyTorch offering useful classes, metrics, or pre-trained networks.  
Link : https://monai.io/


---

# 2. Required trainig data  

The data we use for training is provided by Kaggle, it comes from the Ultrasound Nerve Segmentation challenge. It needs to be downloaded and stored in a folder named *data* to be read automatically by the python script. The data provided by Kaggle needs some renaming before use, this is why you can find a Data_cleaning.py file to launch before the main file execution.

---

# 3. Python file description  

The Classification file is divided into 4 segments which can be compiled separately for a better understanding, and simplify script modifications.

1. Import useful libraries
2. Load Data
3. Define loader classes
4. Train the model



## > Import_useful_libraries  
First segment is simply the import of the libraries presented above.

## > Load_Data
This segment defines a function to simplify the reading of the data, supposing it has been cleand by Data_cleaning.
After this, we load the files and generate images and segmented ouputs for training.

## > Define_loader_classes
Using MONAI framework, we define the transformations to apply to the inputs. This allows us to define classes for our datasets and dataloaders. This is where it is possible to modifiy the train and validation batch size that will be used by the model :
```python
train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
```  
and  
```python
val_loader = DataLoader(val_ds, batch_size=64)
```  

## > Train_the_model
After the settings of our parameters, it is time to start the training. Here, we use a UNet model, but you can change the model or its parameters.
It is possible to change the parameters of the training, like the number of epochs or the learning rate.
At the end of the training, a model will be saved under the name *2D_Nerve_Segmentation.pt* in your repertory.

