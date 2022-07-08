# General Description  

This file contains a classic commented pipeline of medical imaging classification. This algorithm was designed as part of the Kaggle RSNA pneumonia detection challenge, the goal is to diagnose pneumonia using a lung scan. The ouput of the neural network is a label '1' for a pneumonia detected and a label '0' for a healthy patient.  
  
![alt text](Images\Scan_Example.png "Scan Example")
  
On this example, we can see a pneumonia on the right lung of the patient, recognizable by a characteristic opacity on the scan  

---

# 1. Libraries  

We use several libraries that need to be installed upstream :
- Monai
- PyTorch
- Pydicom
- Pandas
- Matplotlib
- Numpy
- Sklearn

## > Monai

Monai proposes a framework to support the creation of a computervision pipeline in the medical field. It is an extension of PyTorch offering useful classes, metrics, or pre-trained networks.  
Link : https://monai.io/

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
4. Train the model
5. Visualize performances



## > Import_useful_libraries  
First segment is simply the import of the libraries presented above.

## > Load_Data
This segment defines different functions to simplify the reading of the data. 
After this, we load the csv file using pandas and generate sample of training and evaluation. It is possible to change the size of the desired sample by modifying this line :
```python
sample_trainingdata = train_labels.groupby('Target', group_keys=False).apply(lambda x: x.sample(4000))
```
The *4000* term designate the number of images per class we want to select for the training, knowing that 20% of them will be used for validation.

## > Define_input_data_and_model_classes
Using MONAI framework, we define the transformations to apply to the inputs. This allows us to define classes for our datasets and dataloaders. This is where it is possible to modifiy the train and validation batch size that will be used by the model :
```python
train_batch_size = 512
val_batch_size = 64
```
After this, we define our model. Here, we use a DenseNet201 pre-trained model, and we re-train the last densblock and the classifier of the network. Regarding the images we want to classify, it might be relevant to change the layers that will be trained.

## > Train_the_model
After the settings of our parameters, it is time to start the training. The Timer class is simply used during training to measure execution time (in seconds). 
The metrics we use here for a binary classification are accuracy, sensitivity and specificity.  
It is possible to change the paramters of the training, like the number of epochs or the learning rate, by modifying these lines :
```python
epochs = 10
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad == True), 0.01)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
```
At the end of the training, a model will be saved under the name *Pneumonia_classification.pt* in your repertory.

## > Visualize_performances
Finally, it is possible to visualize the performances of the model, stored during training in the variable *history*. By commenting or uncommenting lines, you may choose which plots to draw.
![alt text](Images\Accuracy.png "Accuracy")

