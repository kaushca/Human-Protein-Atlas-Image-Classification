
# coding: utf-8

# In[1]:

import os, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import skimage.io
from skimage.transform import resize
#from imgaug import augmenters as iaa
from tqdm import tqdm
import PIL
from PIL import Image
import cv2
from sklearn.utils import class_weight, shuffle
from sklearn.metrics import f1_score
import time
import shutil
from cosine_scheduler import *

# In[2]:

import warnings
warnings.filterwarnings("ignore")
SIZE = 299
batch_size=32
no_of_epochs=100
#THRESHOLD=0.2
#https://www.kaggle.com/c/human-protein-atlas-image-classification/discussion/71648
THRESHOLD=[0.5,0.4,0.4,0.4,0.4,0.4,0.4,0.4,0.2,0.2,0.2,0.4,0.4,0.4,0.4,0.2,0.4,0.4,0.4,0.4,0.2,0.4,0.4,0.4,0.4,0.4,0.4,0.2]

# In[3]:

# Load dataset info
path_to_train = '../input/train/'
train_info = pd.read_csv('../input/train.csv')
path_to_test = '../input/test/'
test_info = pd.read_csv('../input/sample_submission.csv')
validation_rate = 0.2

# In[4]:

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
from sklearn.preprocessing import MultiLabelBinarizer
import torch.nn.functional as F


# In[5]:

label_map={0:"Nucleoplasm",1: "Nuclear membrane",2:"Nucleoli",3:"Nucleoli fibrillar center",   
4:"Nuclear speckles",5: "Nuclear bodies",6: "Endoplasmic reticulum", 7: "Golgi apparatus",   
8: "Peroxisomes",9:"Endosomes",10:"Lysosomes", 11: "Intermediate filaments",  
12:"Actin filaments", 13: "Focal adhesion sites", 14: "Microtubules",   
15: "Microtubule ends",16: "Cytokinetic bridge",17: "Mitotic spindle",   
18: "Microtubule organizing center",19:  "Centrosome", 20:"Lipid droplets",   
21:"Plasma membrane", 22:"Cell junctions", 23:"Mitochondria", 24:"Aggresome",
25:"Cytosol", 26: "Cytoplasmic bodies",27: "Rods & rings"}  

no_of_classes = len(label_map)


# In[ ]:

class proteinImage(Dataset):
    def __init__(self, img_dir, dataframe,train_mode=True,transform=None):
        self.img_dir=img_dir
        self.labels=dataframe
        self.transform=transform
        self.mlb = MultiLabelBinarizer(classes=list(label_map.keys()))
        self.train_mode=train_mode
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        samp=self.labels.Id[idx]
        #if not self.train_mode:
        #    print(samp)
        image_red_ch=Image.open(os.path.join(self.img_dir,samp+"_red.png"))
        image_green_ch=Image.open(os.path.join(self.img_dir,samp+"_green.png"))
        image_blue_ch=Image.open(os.path.join(self.img_dir,samp+"_blue.png"))
        image = Image.merge("RGB",(image_red_ch,image_green_ch,image_blue_ch))
        #image = np.stack((
        #np.array(image_red_ch), 
        #np.array(image_green_ch), 
        #np.array(image_blue_ch)), -1)
        if self.train_mode:
            label = np.array([int(l) for l in self.labels.Target[idx].split(" ")])
            labels_one_hot  = self.mlb.fit_transform([label])
            label = torch.FloatTensor(labels_one_hot)
        else:
            label = self.labels.Predicted[idx] 
        if self.transform:
            image = self.transform(image)

        return [image, label] 


# In[ ]:

train_transform = transforms.Compose([
        transforms.RandomResizedCrop(SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        #imagenet data mean and std assuming the current dataset is not much different
        #if  needed calculate mean and std of a random sample of current dataset
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])
test_transform = transforms.Compose([
        transforms.Resize(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                             std=[0.5, 0.5, 0.5])
    ])


# In[ ]:

validation_total = round(len(train_info) * validation_rate)
val_info=train_info[validation_total:]
val_info=val_info.reset_index()

train_datasets = proteinImage(path_to_train,train_info[:validation_total],train_mode=True,transform=train_transform)
val_datasets=proteinImage(path_to_train,val_info,train_mode=True,transform=test_transform)
trainLoader = DataLoader(train_datasets, batch_size = batch_size, shuffle = True, num_workers=8)
valLoader = DataLoader(val_datasets, batch_size = batch_size, shuffle= True, num_workers=8)
dataloaders={"train":trainLoader, "val":valLoader}


# In[ ]:

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


# In[ ]:

def train_model(model, criterion, optimizer, scheduler, num_epochs=no_of_epochs):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:     
            since_epoch = time.time()
            if phase == 'train':
                if not scheduler == None:
                    scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode
    
            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                # get the inputs
                inputs, labels = data

                inputs = inputs.type(torch.cuda.FloatTensor)
                labels = labels.type(torch.cuda.FloatTensor)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                if phase=="train":
                #https://github.com/pytorch/vision/issues/302
                    outputs,aux = model(inputs) #because inceptionv3 has aux branch
                else:
                    outputs = model(inputs)
                #_, preds = torch.max(outputs.data, 1)
                #print(outputs)
                #print(outputs.type())
                #print(labels.type())
                labels = torch.squeeze(labels)
                loss = criterion(outputs, labels)
                probs = F.sigmoid(outputs)
                predict = probs.cpu().data.numpy()> THRESHOLD 
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    
                # statistics
                running_loss += loss.item()
                #r = (predict == labels.data)  
                #acc = r.float().sum().item()
                #acc = float(acc) / no_of_classes
                #labs = labels.to('cpu')
                #preds = predict.to('cpu')
                #print('Ground Truth: {}'.format(labs))
                #print('Predicted Truth: {}'.format(preds))
                #print('Predicted probs: {}'.format(probs))
                #running_corrects += torch.sum(preds == labels.data)
                fscore=f1_score(labels.data, predict, average='weighted')
                #print(fscore)
                running_corrects +=fscore  

            epoch_loss = running_loss / len(dataloaders[phase])
            #print(running_corrects)
            #print(running_corrects.double())
            epoch_acc = running_corrects / len(dataloaders[phase])

            time_elapsed_epoch = time.time() - since_epoch
            print('{} Loss: {:.4f} Acc: {:.4f} in {:.0f}m {:.0f}s'.format(
                phase, epoch_loss, epoch_acc, time_elapsed_epoch // 60, time_elapsed_epoch % 60))
            
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                is_best=True
                save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer' : optimizer.state_dict(),
                }, is_best)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


def predict_submission(model, submission_load):
    all_preds = []
    model.eval()
    for i, b in enumerate(submission_load):
        #if i % 100: print('processing batch {}/{}'.format(i, len(submission_load)))
        X, _ = b
        if torch.cuda.is_available():
            X = X.cuda()
        pred = model(X)
        all_preds.append(pred.sigmoid().cpu().data.numpy())
    return np.concatenate(all_preds)
        
         
def make_submission_file(sample_submission_df, predictions):
    submissions = []
    for row in predictions:
        subrow = ' '.join(list([str(i) for i in np.nonzero(row)[0]]))
        submissions.append(subrow)
    
    sample_submission_df['Predicted'] = submissions
    sample_submission_df.to_csv('submission.csv', index=None)
    
    return sample_submission_df


# In[6]:

model_ft = models.inception_v3(pretrained='imagenet')
#model_ft.aux_logit=False
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, no_of_classes)
multiGPU = True
use_gpu=True

if torch.cuda.device_count() > 1 and multiGPU:
  print("Using", torch.cuda.device_count(), "GPUs!")
  # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
  model_ft = nn.DataParallel(model_ft)

if use_gpu:
   model_ft.cuda()

# Observe that all parameters are being optimized
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001, momentum=0.9)
optimizer_ft = optim.Adam(model_ft.parameters(),weight_decay=1e-5)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=10, gamma=0.1)
#exp_lr_scheduler = None
exp_lr_scheduler = CosineLRWithRestarts(optimizer_ft,batch_size=batch_size,epoch_size=len(train_info)-validation_total,restart_period=5,t_mult=1.2)

#criterion=nn.MultiLabelSoftMarginLoss()
criterion=nn.BCEWithLogitsLoss()

# In[7]:

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=no_of_epochs)

#out_dir=os.getcwd()
#model_file=os.path.join(out_dir,'model_best.pth.tar')
#checkpoint=torch.load(model_file)
#model_ft.load_state_dict(checkpoint['state_dict'])
#model_ft.cuda()

batch_size=32
submission_dataset=proteinImage(path_to_test,test_info,train_mode=False,transform=test_transform)
submitLoader = DataLoader(submission_dataset, batch_size = batch_size, shuffle = False, num_workers=1)
submission_predictions =predict_submission(model_ft, submitLoader)

# prepare the submission file and 
p = submission_predictions>THRESHOLD

submission_file = make_submission_file(sample_submission_df=test_info,
                     predictions=p)


