# AlexNet
# Description
This project implements AlexNet, a deep learning convolutional neural network, to classify images of cats and dogs. AlexNet was one of the first convolutional networks which demonstrated the outstanding performance of deep learning. This project aims to leverage that architecture to accurately identify whether an image contains a cat or a dog.
# Installation
To get started with this project, follow these steps:
## Install the required dependencies
```
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, optim
from torchvision import transforms, datasets

import zipfile
import shutil
import os
import io
import pandas as pd
```
# Model
The AlexNet model implementation is provided in the alexnet.py file. Make sure this file is in the project directory.
```
class AlexNet(nn.Module):
    def __init__(self, num_classes:int=2):
        super(AlexNet,self).__init__()
        self.convolutional=nn.Sequential(
            nn.Conv2d(3,96,kernel_size=11,stride=4,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #Segunda capa
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),
            #Tercera capa
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            #cuarta capa
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            #Quinta capa
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3,stride=2),

        )
        self.avgpool=nn.AdaptiveAvgPool2d((6,6))
        self.linear = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256*6*6,4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096,2)



    )
    def forward(self, x0:torch.Tensor)-> torch.Tensor:
        x1 = self.convolutional(x0)
        x2 = self.avgpool(x1)
        x3= torch.flatten(x2,1)
        x4 = self.linear(x3)
        return torch.softmax(x4,1)

```
# Labeling the Images
dataset is labeled in a CSV file with image names and categories ("0" for dogs and "1" for cats)
# Training
```
optimizador = optim.Adam(modelo.parameters(),lr=0.0001)
criterio = nn.CrossEntropyLoss()
epocas = 50
errorT = []
for epoca in range(epocas):
  errortotal = 0
  for idx,(image,label) in enumerate(cargar_entrenamiento):
    image,label = image.to(device), label.to(device)
    optimizador.zero_grad()
    prediccion = modelo(image)
    error = criterio(prediccion,label)
    errortotal+=error.item()
    error.backward()
    optimizador.step()
  errortotal = errortotal/(idx+1)
  errorT.append(errortotal)
  print(f'Epoca:{epoca}| error de entrenamiento: {errorT}')



plt.plot(errorT)
```
# Grapic display 
```
xx, yy = next(iter(cargar_testeo))
xx = xx.to(device)
yy = yy.to(device)
with torch.no_grad():
  pre_test = torch.argmax(modelo(xx), axis=1)
fig, axs = plt.subplots(4,4,figsize = (10,10))
for (i,ax) in enumerate(axs.flatten()):
  pic = xx.data[i].cpu().numpy().transpose((1,2,0))
  pic = pic-np.min(pic)
  pic = pic-np.max(pic)
  ax.imshow(pic)
  label = clases.classes[pre_test[i]]
  truec= clases.classes[yy[i]]
  title = f'Prediccion:{label} - True:{truec}'
  titulo = 'g' if truec == label else 'r'
  ax.text(33,67, title, ha = 'center', va ='top',fontweight = 'bold',color = 'k', backgroundcolor=titulo, fontsize = 8)
  ax.axis('off')
plt.tight_layout()
plt.show()
```
