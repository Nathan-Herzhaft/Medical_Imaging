# Folder general description
This folder contains 3 + 1 use-cases of computer vision training algorithms apllied to medicine. It is designed to apprehend and learn medicine-oriented packages for deeplearning in python. Through 3 classical problematics : Classification, Detection and Segmentation, we discover the basis of computervision and learn medical specificities.

# Folder Structure
The file contains 3 commented and documented algorithms, organized according to the same architecture. Additionally, you will find a 4th file for 3D segmentation, whose pipeline is more complex and comes from another opensource challenge.there. Each folder is named after the opensource challenge from which it is taken, and has a unique .py file.  
In each file, it is necessary to download the training datasets, available in opensource, on the challenges websites. Then, you will have to store the dataset in a file named "data", next to the .py file, so that the algorithm automatically finds it. If you choose to store it differently, changes will be necessary in the algorithms to update.  
Here are the link to download the data :  
  
1. [Classification](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
2. [Detection](https://www.kaggle.com/competitions/rsna-pneumonia-detection-challenge/data)
3. [Segmentation2D](https://www.kaggle.com/competitions/ultrasound-nerve-segmentation/data)  

Additional. [Segmentation3D](https://covid-segmentation.grand-challenge.org/Download/)

  
Next to the .py files, you can find a README file per use-case, commenting the different steps of the algorithm to learn how to use and modify it.  

# Use-cases description
Here is the ideal order to explore these use cases :

1. Classification
2. Detection
3. Segmentation2D
Additional. Segmentation3D
  
## > Classification
This algorithm addresses a typical problem of classification for diagnosis assistance. The objective is to build a neural network allowing the diagnosis of a pneumonia (Classification 0 or 1). The goal of this pipeline is to familiarize with deeplearning frameworks in Python, and in particular with *monai*, a framework specialized in computer vision for the medical field. 

## > Detection
After classification, we turn to a detection problem. Using the same dataset, we seek this time to localize pneumonia in addition to diagnosing it. The particularity is that this detection pipeline does not use specialized libraries for medical applications. Indeed, detection is not a frequent problem in medicine, because the objects we are trying to detect are more suited to segmentation problems. However, this pipeline is useful to understand the PyTorch library, on which the various specialized libraries are based.  

## > Segmentation 2D
The last typical problem to study is segmentation. Still based on Kaggle, we seek to segment nerve collection on ultrasound images. The pipeline is not complicated and is easily applicable to other types of problems by changing the first preprocessing functions. This example has a slight peculiarity: it is necessary to perform transformations on the input data before running the program. You will find for that the file Data_cleaning.py which will format correctly the files.

## > Segmentation 3D
The body of this git focuses on 2D algorithms based on Kaggle. Therefore you will find an additional example for the 3D case, which is useful to get familiar with the preprocess and the adapted transformations. This dataset is no longer based on a Kaggle challenge, but on the Grand Challenge for lung lesion segmentation. The goal is to locate the lung lesions on a 3D CT scan, using monai framework.

# Medical Imaging
These algorithms are based on medical data, whose formats are specific and are read by specialized python libraries.  
  
![alt text](Images\Scan_Example.png "Scan Example")  
Lung Scan opened with pydicom library