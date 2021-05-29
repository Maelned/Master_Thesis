import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import pickle

# os.chdir("/home/ubuntu/Implementation_Mael")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
# dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\"
dataset = "/mnt/data/Dataset/ISIC2018V2/"
training_dataset = dataset + "Training/"
validation_dataset = dataset + "Validation/"
test_dataset = dataset + "Test/"

loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)
Healthy_or_not = [0,0,1,1,0,1,1]
epsilon = 2/255.
labels = {"akiec" : [[1.,0.,0.,0.,0.,0.,0.]],
            "bcc" : [[0.,1.,0.,0.,0.,0.,0.]],
            "bkl" : [[0.,0.,1.,0.,0.,0.,0.]],
            "df" :  [[0.,0.,0.,1.,0.,0.,0.]],
            "mel" : [[0.,0.,0.,0.,1.,0.,0.]],
            "nv"  : [[0.,0.,0.,0.,0.,1.,0.]],
            "vasc": [[0.,0.,0.,0.,0.,0.,1.]]
        }
with open("./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3.pkl", "rb") as f:
    cm_InceptionV3= pickle.load(f)


def modif_cm(confusion_matrix):
    new_confusion_matrix = []

    Healthy= 0
    Healthy_as_Cancerous = 0
    Healthy_as_Healthy = 0
    Cancerous = 0
    Cancerous_as_Healthy = 0
    Cancerous_as_Cancerous = 0

    for label in range(len(confusion_matrix)):
        row = confusion_matrix[label, :]
        if Healthy_or_not[label]:
            for i in range(len(row)):
                if i == label:
                    Healthy += row[i]
                else:
                    if Healthy_or_not[i]:
                        Healthy_as_Healthy += row[i]
                    else:
                        Healthy_as_Cancerous += row[i]
        else:
            for i in range(len(row)):
                if i == label:
                    Cancerous += row[i]
                else:
                    if Healthy_or_not[i]:
                        Cancerous_as_Healthy += row[i]
                    else:
                        Cancerous_as_Cancerous += row[i]
        List_Healthy = [Healthy,Healthy_as_Healthy,Healthy_as_Cancerous]
        List_Cancerous = [Cancerous,Cancerous_as_Healthy,Cancerous_as_Cancerous]
        new_confusion_matrix = [List_Healthy,List_Cancerous]
        new_confusion_matrix = np.array(new_confusion_matrix)
    return new_confusion_matrix


print(cm_InceptionV3)
new_cm_InceptionV3 = modif_cm(cm_InceptionV3)
print(new_cm_InceptionV3)

import matplotlib.pyplot as plt

fig, ax =plt.subplots(1,1)

column_labels=["Correctly classified", "Classified as Healthy", "Classified as Cancerous"]
row_labels = ["Healthy","Cancerous"]
ax.axis('tight')
ax.axis('off')
ax.table(cellText=new_cm_InceptionV3,colLabels=column_labels,rowLabels=row_labels,loc="center")

plt.show()
