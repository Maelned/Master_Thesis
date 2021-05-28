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
model = load_model("./Saves/Models/InceptionV3_v3.h5")

loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

epsilon = 2/255.
labels = {"akiec" : [[1.,0.,0.,0.,0.,0.,0.]],
            "bcc" : [[0.,1.,0.,0.,0.,0.,0.]],
            "bkl" : [[0.,0.,1.,0.,0.,0.,0.]],
            "df" :  [[0.,0.,0.,1.,0.,0.,0.]],
            "mel" : [[0.,0.,0.,0.,1.,0.,0.]],
            "nv"  : [[0.,0.,0.,0.,0.,1.,0.]],
            "vasc": [[0.,0.,0.,0.,0.,0.,1.]]
        }


test_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

test_ds = test_datagen.flow_from_directory(
    test_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=1,
    shuffle=False,
    seed=False,
    interpolation="bilinear",
    follow_links=False)


Y_pred = model.predict_generator(test_ds, steps=test_ds.samples / 1)
y_pred = np.argmax(Y_pred, axis=1)

cm = confusion_matrix(test_ds.classes, y_pred)
cm = np.around(cm, 2)

name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3.pkl"
with open(name_cm, 'wb') as f:
    pickle.dump(cm, f)
