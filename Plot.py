import numpy as np
import os
import pickle
import tensorflow as tf
import seaborn as sns
import pandas as pd
from matplotlib.pyplot import figure
import operator
from keras import regularizers
from keras import layers
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight
from keras.optimizers import RMSprop, SGD, Adam
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import accuracy_score
from keras.metrics import categorical_accuracy
from collections import Counter, OrderedDict

data_dir = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\HAM10K\\"
data_dir5 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabels\\5% nv - bkl\\"
data_dir10 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabels\\10% nv - bkl\\"
data_dir15 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabels\\15% nv - bkl\\"
data_dir20 = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ModifiedLabels\\20% nv - bkl\\"
label = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']

datagen = ImageDataGenerator(
    rescale=1. / 255.,
)

dataset5 = datagen.flow_from_directory(
    data_dir5,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset10 = datagen.flow_from_directory(
    data_dir10,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset15 = datagen.flow_from_directory(
    data_dir15,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset20 = datagen.flow_from_directory(
    data_dir20,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

dataset = datagen.flow_from_directory(
    data_dir,
    target_size=(224, 224),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=32,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)



Labels_count = Counter(dataset5.classes)
Label_sorted = sorted(Labels_count.items(), key=operator.itemgetter(1),reverse=True)
x,y = zip(*Label_sorted)

print(x,y)
x = list(x)
Modified = [0,335,0,0,0,0,0]
y = list(y)
y[1] = y[1] - 335
a = 0
for i in x:
    x[a] = label[i]
    a += 1

print(x)
index = range(len(x))
index2 = range(len(y))
plt.ylim([0,6800])
my_colors = ["tab:blue","tab:orange","tab:green","tab:red","tab:purple","tab:brown","tab:pink"]
# s = pd.Series(y, index=x)
#
# s.plot(kind='bar',rot=0)
plt.bar(x,y,color=my_colors,width=0.5)
plt.bar(x,Modified,color = "tab:blue", bottom=y,width=0.5)
plt.title('Classes distribution with 5% of NV in BKL')
plt.show()
