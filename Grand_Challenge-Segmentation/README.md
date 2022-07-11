# General Description  

This file contains a classic commented pipeline of medical imaging segmentation. This algorithm was designed as part of the Grand Challenge - COVID 19 lung CT lesion segmentation, the goal is to diagnose and locate lung lesions using lung scans.
This pipeline does not use a specialized library python for medical deeplearning, since detection is rarely used in medicine. Refer to Classification or Segmentation examples to use these libraries. 

---

# 1. Libraries  

We use several libraries and packages that need to be installed upstream :
- Monai
- PyTorch
- Numpy

## > Monai

Monai proposes a framework to support the creation of a computervision pipeline in the medical field. It is an extension of PyTorch offering useful classes, metrics, or pre-trained networks.  
Link : https://monai.io/

---

# 2. Required trainig data  

The data we use for training is provided by Grand Challenge, it comes from the Grand Challenge - COVID 19 lung CT lesion segmentation. It needs to be downloaded and stored in a folder named *data* to be read automatically by the python script.

---

# 3. Python file description  

The Segmentation file is divided into 5 segments which can be compiled separately for a better understanding, and simplify script modifications.

1. Import useful libraries
2. Define transformations
3. Define model and loss function
4. Define training and inference functions
5. Train the model



## > Import useful libraries  
First segment is simply the import of the libraries presented above.

## > Define transformations
Using monai framework, we define the transformations to apply to the inputs.

## > Define model and loss function
Using monai framework, we define our model.Here, we use a Basic UNet pre-trained model.

## > Define training and inference functions
Initialize the settings of the training through training and inference functions. It is possible to change the parameters of the training, like the number of epochs or the learning rate.
  
![alt text](Images\Box_examples.png "Box Examples")  
  
## > Train the_model
After the settings of our parameters, it is time to start the training. 