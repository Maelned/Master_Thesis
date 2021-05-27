import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix

import pickle
#
# with open("/home/ubuntu/Implementation_Kentin/Perf/CM_RTInception_POSTFGSM.pkl", 'rb') as f:
#     cm_InceptionV3_FGSM = pickle.load(f)
#
# with open("/home/ubuntu/Implementation_Kentin/Perf/CM_RTInception_FGSMFlipped_Compressed.pkl", 'rb') as f:
#     cm_InceptionV3_FGSM_Defended = pickle.load(f)
#
# print(cm_InceptionV3_FGSM)
# print(cm_InceptionV3_FGSM_Defended)
#
#
# name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_method2.pkl"
# with open(name_cm, 'wb') as f:
#     pickle.dump(cm_InceptionV3_FGSM, f)
#
# name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_InceptionV3_FGSM_FlippedCompressed.pkl"
# with open(name_cm, 'wb') as f:
#     pickle.dump(cm_InceptionV3_FGSM_Defended, f)

# os.chdir("/home/ubuntu/Implementation_Mael")
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
dataset = "E:\\NTNU\\TTM4905 Communication Technology, Master's Thesis\\Code\\Dataset\\ISIC2018V2\\"
# dataset = "/mnt/data/Dataset/ISIC2018V2/"
training_dataset = dataset + "Training/"
validation_dataset = dataset + "Validation/"
test_dataset = dataset + "Test/"
model = load_model("./Saves/Models/Retrained_model_v3_UAP_5epoch_8times.h5")
# model = load_model("/home/ubuntu/Implementation_Kentin/Perf/ResNetV2.h5")

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

#
# Y_pred = model.predict_generator(test_ds, steps=test_ds.samples / 1)
# y_pred = np.argmax(Y_pred, axis=1)
#
# cm = confusion_matrix(test_ds.classes, y_pred)
# cm = np.around(cm, 2)
#
# name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_Resnet_FGSM.pkl"
# with open(name_cm, 'wb') as f:
#     pickle.dump(cm, f)

def create_adversarial_pattern(input_image,input_label,model):
    with tf.GradientTape() as tape:
        input_image = tf.convert_to_tensor(input_image, dtype=tf.float32)
        # explicitly indicate that our image should be tacked for
        # gradient updates
        tape.watch(input_image)
        # use our model to make predictions on the input image and
        # then compute the loss
        pred = model(input_image)
        loss = loss_object(input_label,pred)
        # calculate the gradients of loss with respect to the image, then
        # compute the sign of the gradient
        gradient = tape.gradient(loss, input_image)
        signedGrad = tf.sign(gradient)
        return signedGrad

preds = []
nb_img = 0
for e in range(len(test_ds)):
    i = next(test_ds)
    image = i[0]
    label = i[1]
    adv_noise = create_adversarial_pattern(image,label,model)
    # construct the image adversary
    img_adv = (image + (adv_noise * epsilon))
    # img_adv= tf.clip_by_value(img_adv, -1, 1)
    prediction = model.predict(img_adv)
    preds.append(prediction[0])

preds = list(preds)
preds = np.argmax(preds, axis=1)
cm_adv = confusion_matrix(test_ds.classes, preds)
cm_adv = np.around(cm_adv, 2)
print(cm_adv)

name_cm = "./Saves/ConfusionMatrixes/ConfusionMatrix_RetrainedInceptionV3UAP_FGSM.pkl"
with open(name_cm, 'wb') as f:
    pickle.dump(cm_adv, f)