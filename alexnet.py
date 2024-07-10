# -*- coding: utf-8 -*-
"""AlexNet.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1DmhjLH7o4Ba1IHoRrIvL6SAqZQ99B9Dm
"""

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

device = ('cuda' if torch.cuda.is_available() else "CPU") #Usar gpu o cpu

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



modelo = AlexNet()
modelo = modelo.to(device)

from google.colab import files
uploader = files.upload()

datos = zipfile.ZipFile('baseDatosDogCat.zip')
datos.extractall("perros_gatos/")

datos = zipfile.ZipFile(io.BytesIO(uploader['baseDatosDogCat.zip']))
datos.extractall("perros_gatos/")

root = '/content/perros_gatos/baseDatosDogCat/data_base_dog_cat'
img_list=os.listdir(root)
print(len(img_list))

basedatos=pd.read_csv('/content/perros_gatos/baseDatosDogCat/lista_dog_cat.csv')
basedatos=basedatos[['Nombre','Categoria']]
print(basedatos)

!rm -rf datos
!mkdir datos && mkdir datos/perros && mkdir datos/gatos

s0=0
s1=0
num=1000
for i,(_,i_row)in enumerate(basedatos.iterrows()):
  if s0 < num:
    if(i_row['Categoria'] == 1):
      s0+=1
      shutil.copyfile('/content/perros_gatos/baseDatosDogCat/data_base_dog_cat/'+ i_row['Nombre'],'datos/perros/' + i_row['Nombre'])
  if(s1 < num):
    if(i_row['Categoria']==0):
      s1+=1
      shutil.copyfile('/content/perros_gatos/baseDatosDogCat/data_base_dog_cat/'+ i_row['Nombre'],'datos/gatos/' + i_row['Nombre'])
  if(s0 == num and s1 == num):
    break

img_list = os.listdir('datos/perros/')
img_list.extend(os.listdir('datos/gatos/'))
!rm -rf 'datos/.ipynb_checkpoints/'
transformar = transforms.Compose([transforms.ToTensor(),transforms.Resize((224,224)),transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229,0.224,0.225])])
clases = datasets.ImageFolder('datos',transform = transformar)
entrenamiento, testeo = torch.utils.data.random_split(clases,[int(len(img_list)*0.8),len(img_list)-int(len(img_list)*0.8)])
cargar_entrenamiento = torch.utils.data.DataLoader(entrenamiento,batch_size=32,shuffle=True)
cargar_testeo = torch.utils.data.DataLoader(testeo,batch_size=16,shuffle=True)

x,y = next(iter(cargar_entrenamiento))

fig, axs = plt.subplots(4,4,figsize= (10,10))
for (i,ax) in enumerate(axs.flatten()):
  pic= x.data[i].numpy().transpose((1,2,0))
  pic = pic-np.min(pic)
  pic= pic/np.max(pic)
  label= clases.classes[y[i]]
  ax.imshow(pic)
  ax.text(0,0,label,ha='left',va='top',fontweight='bold',color='k',backgroundcolor='y')
  ax.axis('off')
plt.tight_layout()
plt.show



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

plt.plot(errorT)

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