import os, random
import numpy as np
from os import listdir
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator

new_model = load_model("Saves/Models/InceptionV3.h5")
loss_object = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.NONE)

def create_adversarial_pattern(input_image,input_label):
    with tf.GradientTape() as tape:

        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(input_image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = new_model(input_image)
        loss = loss_object(input_label,pred)
        # calculate the gradients of loss with respect to the image, then
        # compute the sign of the gradient
        gradient = tape.gradient(loss, input_image)
        signedGrad = tf.sign(gradient)

        return signedGrad

Dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\Dataset_Adversarial_Samples\\"
Dataset_ref = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\"

eps = 2/255.0
amount = [0.1,0.2,0.3,0.4,0.5]
experiment = listdir(Dataset)
experiment = sorted(experiment)
classes = ["akiec","bcc","bkl","df","mel","nv","vasc"]
label = [[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0],[0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]]
set = ["\\Training\\","\\Validation\\"]
print("Starting loops")
for i in range(len(amount)):
    print("Current experiment : ",experiment[i])
    for current_set in set:
        print("Current set : ",current_set)
        for current_class in classes:
            print("Current directory : ", current_class)
            num_ref = len(listdir(Dataset_ref + current_set + current_class))
            index = classes.index(current_class)
            current_label = [label[index]]
            # current_dataset = Dataset + current_set + current_class
            am = amount[i] * num_ref
            dir = experiment[i]
            current_dir = Dataset + dir + current_set + current_class
            filenames = random.sample(os.listdir(current_dir), int(am))

            for fname in filenames:

                current_img = tf.keras.preprocessing.image.load_img(os.path.join(current_dir,fname),target_size=(224,224))
                current_img = tf.keras.preprocessing.image.img_to_array(current_img)
                current_img = current_img.reshape([1, 224, 224, 3])

                adv_noise = create_adversarial_pattern(current_img, current_label)
                adv_img = (current_img + (adv_noise * eps))
                adv_img = tf.keras.preprocessing.image.array_to_img(adv_img[0])
                adv_img = adv_img.resize((600,450))
                tf.keras.preprocessing.image.save_img(current_dir + "\\adv_" + fname.strip(),adv_img)
                # cv2.imwrite(current_dir + "\\adv_" + fname.strip(), adv_img)

