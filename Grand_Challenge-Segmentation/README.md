# General Description  

This file contains a classic commented pipeline of medical imaging segmentation. This algorithm was designed as part of the Grand Challenge - COVID 19 lung CT lesion segmentation, the goal is to diagnose and locate lung lesions using lung scans.

![alt text](Images\Scan_Example.png "Scan Example")

---

# 1. Libraries  

We use several libraries and packages that need to be installed upstream :
- Monai
- PyTorch
- Numpy

## > Monai

Monai proposes a framework to support the creation of a computervision pipeline in the medical field. It is an extension of PyTorch offering useful classes, metrics, or pre-trained networks.  
Link : https://monai.io/  
Command for installing :  
pip install "git+https://github.com/Project-MONAI/MONAI#egg=monai[nibabel,ignite,tqdm]"

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
Using monai framework, we define the transformations to apply to the inputs. You can find the description of each transformation on the monai website.

## > Define model and loss function
Using monai framework, we define our neural network and inferer. Here, we use a Basic UNet pre-trained model. For the Loss function, a basic crosse entropy function is used.

## > Define training and inference functions
Initialize the settings of the training through training and inference functions. It is possible to change the parameters of the training, like the number of epochs or the learning rate.

You can make changes on the model paramaters by modifying these lines :
```python
batch_size = 2
```
and
```python
max_epochs, lr, momentum = 500, 1e-4, 0.95
```
  
## > Train the_model
After the settings of our parameters, it is time to start the training. This segment use parser to define a command starting the training. Therefore, this algorithm is started using the terminal.  

To start the training, type this in the terminal :  
  
python Grand_Challenge-Segmentation/Segmentation.py train --data_folder "Grand_Challenge-Segmentation/data/Train" --model_folder "runs"  
  
During training, the top three models will be selected based on the per-epoch validation and stored at --model_folders.  
  
Change the name of the file or the data_folder argument if you customized your files differently.