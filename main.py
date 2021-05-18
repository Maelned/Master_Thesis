import numpy as np
import os
import tensorflow as tf
from keras import regularizers
from keras import layers
from keras.models import Model
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight
from keras.optimizers import Adam,SGD
from keras.callbacks import LearningRateScheduler
from sklearn.metrics import accuracy_score
from keras.metrics import categorical_accuracy

# ***************************** NEW CODE *********************************

physical_device = tf.config.experimental.list_physical_devices('GPU')
print(f'Device found : {physical_device}')

# Location of this program
os.chdir("/home/ubuntu/Implementation_Mael")
dataset = "/mnt/data/Dataset/ISIC2018V2/"
training_dataset = dataset + "Training/"
validation_dataset = dataset + "Validation/"
# A path to the folder where the rearranged images are stored:
# as rearranged, it means :
# - HAM10K
#   -ClassA
#       -images
#   -ClassB
#       -images
#   .
#   .
#   .

label = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
classes = ['actinic keratoses', 'basal cell carcinoma', 'benign keratosis-like lesions',
           'dermatofibroma', 'melanoma', 'melanocytic nevi', 'vascular lesions']

# different parameters for the model
batch_size = 32
nb_epochs = 50

# **************** Dataset Creation ********************

train_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=180,  # randomly rotate images in the range (degrees, 0 to 180)
    zoom_range=0.1,  # Randomly zoom image
    width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=False,  # randomly flip images
    vertical_flip=False,
)

val_datagen = ImageDataGenerator(
    rescale=1. / 255.,
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
)

train_ds = train_datagen.flow_from_directory(
    training_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)

val_ds = val_datagen.flow_from_directory(
    validation_dataset,
    target_size=(299, 299),
    color_mode="rgb",
    classes=None,
    class_mode="categorical",
    batch_size=batch_size,
    shuffle=True,
    seed=False,
    interpolation="bilinear",
    follow_links=False)


class_weights = class_weight.compute_class_weight("balanced",
                                                  np.unique(train_ds.classes),
                                                  train_ds.classes)
class_weights = {i: class_weights[i] for i in range(7)}

# **************** Model Creation (import the Inception V3 and perform transfer learning) ********************
pre_trained_model = InceptionV3(input_shape=(299, 299, 3), include_top=False, weights="imagenet")

# for layer in pre_trained_model.layers:
#   layer.trainable = False

# add a global spatial average pooling layer

x = pre_trained_model.output
x = layers.GlobalAveragePooling2D()(x)

# # add a fully-connected layer
x = layers.Dropout(0.7)(x)
x = layers.Dense(units=512,kernel_regularizer= regularizers.l1(1e-3),activation='relu')(x)
x = layers.Dropout(0.5)(x)
# and a fully connected output/classification layer
x = layers.Dense(7,kernel_regularizer= regularizers.l1(1e-3),activation="softmax")(x)
# x = layers.Activation(activation='softmax')(x)
# create the full network so we can train on it
model = Model(pre_trained_model.input, x)

def scheduler(epoch, lr):
  if epoch == 40:
    return lr / 10
  elif epoch == 45:
    return lr / 10
  else:
    return lr
learning_rate_reduction = LearningRateScheduler(scheduler)

model.compile(optimizer=SGD(lr=1e-3,momentum=0.9), loss="categorical_crossentropy", metrics=[categorical_accuracy])

history = model.fit_generator(
    train_ds,
    steps_per_epoch=train_ds.samples // batch_size,
    validation_data=val_ds,
    validation_steps=val_ds.samples // batch_size,
    epochs=nb_epochs,
    initial_epoch=0,
    verbose=2,
    class_weight=class_weights,
    workers=8,
    callbacks=[learning_rate_reduction]
)

# ******************* Printing Confusion Matrix ***************
model.evaluate_generator(val_ds, val_ds.samples // batch_size, verbose=2)

Y_pred = model.predict_generator(val_ds, steps=val_ds.samples / batch_size)
y_pred = np.argmax(Y_pred, axis=1)

accuracy_scr = accuracy_score(val_ds.classes, y_pred)

print("ACCURACY SCORE = ", accuracy_scr)

np.save('./pythonProject1/Saves/Hitsory/history_InceptionV3_v1.npy', history.history)
model.save("./pythonProject1/Saves/Models/InceptionV3_v1.h5")
